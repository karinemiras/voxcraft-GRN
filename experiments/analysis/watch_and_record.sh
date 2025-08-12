# run this script from the root (revolve folder): ./experiments/default_study/run-analysis.sh PARAMSFILE

DIR="$(dirname "${BASH_SOURCE[0]}")"
study_path="$(basename $DIR)"

if [ $# -eq 0 ]
  then
     params_file=$DIR/paramsdefault.sh
  else
    params_file=$1
fi

source $params_file


file="${outputs_path}/${study}/analysis/video_bests.mpg";

printf " \n resimulating ..."
#screen -d -m -S ${study}_videos ffmpeg -f x11grab -r 25 -i :1 -qscale 0 $file;
python3 experiments/${study_path}/watch_robots.py $study $experiments $tfs $watchruns $generations $outputs_path;

pkill -f ${study}_videos
pkill -f ${study}_loop

printf " \n finished!"