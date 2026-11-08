#!/bin/bash
#SBATCH --job-name=lammps
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=lmp_%j.out
#SBATCH --error=lmp_%j.err

set -euo pipefail
source /home/stud17/miniconda3/etc/profile.d/conda.sh
conda activate lammps-kim
cd "${SLURM_SUBMIT_DIR:-$PWD}"

INPUT=${1:-two_phase.in}
LMP_BIN=${LMP_BIN:-lmp}
MPIRUN=${MPIRUN:-mpirun}
DIR=cells_openff/610381

# ===================== RUN PARAMETERS =====================
TEMPS=(350 400 500 600)   # K: one independent run per value
PRESSURES=(1)          # atm: one independent run per value (cross product with TEMPS)
MAX_PER=8                 # max MPI ranks per run (~250+ atoms per rank)
RUN_EQUIL=${RUN_EQUIL:-1} # 0 = skip crystal+liquid stages (if *_eq.data exist)
NMOL=160
# ==========================================================

mkdir -p "$DIR"
INPUT_NAME=$(basename "$INPUT")
cp -f "$INPUT" "$DIR/$INPUT_NAME"
INPUT_RUN="$DIR/$INPUT_NAME"

# Crystal file and molecule count are taken from <DIR>/<name>_n<N>.data
STEM=$(basename "$DIR")
CRYST="$DIR/${STEM}_n${NMOL}.data"
if [[ ! -f "$CRYST" ]]; then
    echo "Missing $CRYST" >&2
    exit 1
fi

# Build (T, P) combinations
COMBOS=()
for T in "${TEMPS[@]}"; do
    for P in "${PRESSURES[@]}"; do
        COMBOS+=("${T}:${P}")
    done
done
NC=${#COMBOS[@]}
if (( NC < 1 )); then echo "No T/P combinations" >&2; exit 1; fi

# Ranks per run
PER=$(( SLURM_NTASKS / NC ))
if (( PER > MAX_PER )); then PER=$MAX_PER; fi
if (( PER < 1 )); then PER=1; fi

# How many runs to launch simultaneously
MAX_JOBS=$(( SLURM_NTASKS / PER ))
if (( MAX_JOBS < 1 )); then MAX_JOBS=1; fi
if (( MAX_JOBS > NC )); then MAX_JOBS=$NC; fi

# Available cores (Slurm-aware), split into groups of PER
read -ra CPUS <<< "$(python3 -c "
import sys
r=[]
for x in sys.argv[1].split(','):
    a,_,b=x.partition('-'); r.extend(range(int(a),int(b or a)+1))
print(*r)" "$(taskset -pc $$ | sed 's/.*: //')")"

AVAIL=${#CPUS[@]}
if (( AVAIL < PER )); then
    echo "Cores ${AVAIL} < ${PER} (need at least one run worth)" >&2
    exit 1
fi
if (( MAX_JOBS * PER > AVAIL )); then
    MAX_JOBS=$(( AVAIL / PER ))
fi

echo "Job ${SLURM_JOB_ID}: ${NC} combos (T x P), ${PER} ranks each, up to ${MAX_JOBS} in parallel; $(date)"
echo "Temperatures: ${TEMPS[*]} K; pressures: ${PRESSURES[*]} atm"
echo "Crystal: ${CRYST}, molecules: ${NMOL}"

run_TP() {   # $1 = T, $2 = P, $3 = list of cores for this run
    local T=$1 P=$2 cpus=$3
    local tag="T${T}_P${P}"
    local pairs=("joined:joined")
    if [[ "$RUN_EQUIL" == "1" ]]; then pairs=("crystal liquid:equil" "joined:joined"); fi
    local pair st name
    for pair in "${pairs[@]}"; do
        st=${pair%%:*}; name=${pair##*:}
        # taskset pins the run; --bind-to none stops mpirun from rebinding
        if ! taskset -c "$cpus" "${MPIRUN}" -np "$PER" --bind-to none \
            "${LMP_BIN}" -in "$INPUT_RUN" \
                -var T_run "$T" -var P "$P" -var dir "$DIR" \
                -var cryst_file "$CRYST" -var n_mol_cryst "$NMOL" -var stage $st \
            -log "$DIR/log.${SLURM_JOB_ID}.${tag}.${name}" \
            -screen "$DIR/screen.${SLURM_JOB_ID}.${tag}.${name}" -nonbuf \
            2> "$DIR/err.${SLURM_JOB_ID}.${tag}.${name}"; then
            echo "[$tag] ${name}: ERROR. stderr:"
            tail -n 15 "$DIR/err.${SLURM_JOB_ID}.${tag}.${name}" || true
            echo "[$tag] last screen lines:"
            tail -n 10 "$DIR/screen.${SLURM_JOB_ID}.${tag}.${name}" || true
            return 1
        fi
    done
    echo "[$tag] done: $(date)"
}

# Launch combos in batches of MAX_JOBS
fail=0
for (( base=0; base<NC; base+=MAX_JOBS )); do
    pids=()
    for (( k=base; k<base+MAX_JOBS && k<NC; k++ )); do
        combo=${COMBOS[$k]}
        T=${combo%%:*}; P=${combo##*:}
        slot=$(( k - base ))
        cpus=$(IFS=,; echo "${CPUS[*]:$((slot*PER)):PER}")
        run_TP "$T" "$P" "$cpus" &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    if (( fail != 0 )); then
        echo "Batch failed, stopping further batches. Already-computed (T,P) can be resumed with RUN_EQUIL=0 and -var stage joined."
        break
    fi
done

echo "Finished: $(date), code $fail"
exit $fail
