#!/bin/bash
#SBATCH --job-name=lammps
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=lmp_%j.out
#SBATCH --error=lmp_%j.err

set -euo pipefail
source /home/stud17/miniconda3/etc/profile.d/conda.sh
conda activate lammps-kim
cd "${SLURM_SUBMIT_DIR:-$PWD}"

INPUT=${1:-two_phase.in}
LMP_BIN=${LMP_BIN:-lmp}
MPIRUN=${MPIRUN:-mpirun}
DIR=cells_openff/131762

# ===================== ПАРАМЕТРЫ ЗАПУСКА =====================
TEMPS=(320 360 380 400)   # K: по одному независимому запуску на значение
MAX_PER=8                 # макс. MPI-рангов на запуск (~250+ атомов на ранг)
# ==============================================================

mkdir -p "$DIR"
NT=${#TEMPS[@]}
PER=$(( SLURM_NTASKS / NT ))
if (( PER > MAX_PER )); then PER=$MAX_PER; fi
if (( PER < 1 )); then echo "Температур больше, чем задач" >&2; exit 1; fi

# список доступных ядер (учитывает ограничения Slurm), раскладываем по группам
read -ra CPUS <<< "$(python3 -c "
import sys
r=[]
for x in sys.argv[1].split(','):
    a,_,b=x.partition('-'); r.extend(range(int(a),int(b or a)+1))
print(*r)" "$(taskset -pc $$ | sed 's/.*: //')")"
if (( ${#CPUS[@]} < NT*PER )); then echo "Ядер ${#CPUS[@]} < ${NT}x${PER}" >&2; exit 1; fi

echo "Job ${SLURM_JOB_ID}: температуры ${TEMPS[*]}, по ${PER} рангов; $(date)"

run_T() {   # $1 = T, $2 = список ядер для этой температуры
    local T=$1 cpus=$2 tag="T$1" st name
    for pair in "crystal liquid:equil" "joined:joined"; do
        st=${pair%%:*}; name=${pair##*:}
        # taskset держит этот запуск на своих ядрах; --bind-to none, чтобы mpirun не перепривязывал
        if ! taskset -c "$cpus" "${MPIRUN}" -np "$PER" --bind-to none \
            "${LMP_BIN}" -in "$INPUT" -var T_run "$T" -var dir "$DIR" -var stage $st \
            -log "$DIR/log.${SLURM_JOB_ID}.${tag}.${name}" \
            -screen "$DIR/screen.${SLURM_JOB_ID}.${tag}.${name}" -nonbuf; then
            echo "[$tag] ${name}: ОШИБКА, последние строки:"
            tail -n 15 "$DIR/screen.${SLURM_JOB_ID}.${tag}.${name}" || true
            return 1
        fi
    done
    echo "[$tag] завершено: $(date)"
}

pids=()
for i in "${!TEMPS[@]}"; do
    cpus=$(IFS=,; echo "${CPUS[*]:$((i*PER)):PER}")
    run_T "${TEMPS[$i]}" "$cpus" &
    pids+=($!)
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "Finished: $(date), код $fail"
exit $fail
