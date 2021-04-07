#!/bin/bash

models_var=HLS4ML_KERAS_MODELS
models_file=
exec=echo
py=3
dir=

function print_usage {
   echo "Usage: `basename $0` [OPTION]"
   echo ""
   echo "Reads the model names from the ${models_var} environment variable"
   echo "or provided file name and optionally starts the conversion."
   echo ""
   echo "Options are:"
   echo "   -f FINENAME"
   echo "      File name to read models from. If not specified, reads from ${models_var}"
   echo "      environment variable."
   echo "   -x"
   echo "      Execute the commands instead of just printing them."
   echo "   -p 2|3"
   echo "      Python version passed to keras-to-hls script (2 or 3)."
   echo "   -d DIR"
   echo "      Output directory passed to keras-to-hls script."
   echo "   -h"
   echo "      Prints this help message."
}

while getopts ":f:xp:d:h" opt; do
   case "$opt" in
   f) models_file=$OPTARG
      ;;
   x) exec=eval
      ;;
   p) py="-p $OPTARG"
      ;;
   d) dir="-d $OPTARG"
      ;;
   h)
      print_usage
      exit
      ;;
   :)
      echo "Option -$OPTARG requires an argument."
      exit 1
      ;;
   esac
done

if [ -z ${models_file} ]; then #nao tem modelo file entao
   if [ -z ${!models_var+x} ] ; then #nao tem modelo defeault entao
      echo "No file provided and ${models_var} variable not set. Nothing to do." #erro
      exit 1 #termina o codigo
   else
      IFS=";" read -ra model_line <<< "${!models_var}" #le o arquivo defeaut 
   fi
else
   readarray model_line < "${models_file}" #le o arquivo enviado
fi

for line in "${model_line[@]}"
do
   params=("" "" "" "" "" "" "" "" "")
   if [[ ${line} = *[![:space:]]* ]] && ! [[ "${line}" = \#* ]] ; then
      IFS=" " read -ra model_def <<< "${line}"
      for (( i=1; i<"${#model_def[@]}"; i++ ));
      do
         if [[ "${model_def[$i]}" == f:* ]] ; then params[0]="-f ${model_def[$i]:2} "; fi
         if [[ "${model_def[$i]}" == c:* ]] ; then params[1]="-c ${model_def[$i]:2} "; fi
         if [[ "${model_def[$i]}" == b:* ]] ; then params[2]="-b ${model_def[$i]:2} "; fi
         if [[ "${model_def[$i]}" == r:* ]] ; then params[3]="-r ${model_def[$i]:2} "; fi
         if [[ "${model_def[$i]}" == s:* ]] ; then params[4]="-g ${model_def[$i]:2} "; fi
         if [[ "${model_def[$i]}" == t:* ]] ; then params[5]="-t ${model_def[$i]:2} "; fi
         if [[ "${model_def[$i]}" == io:s ]] ; then params[6]="-s "; fi
      done
      params[7]=${model_def[0]}

      cmd="./keras-to-hls.sh -p 3 ${dir} ${params[0]}${params[1]}${params[2]}${params[3]}${params[4]}${params[5]}${params[6]}${params[7]}" #eco é o print do .sh
      echo $cmd

      ${exec} "${cmd}"
   fi
done

#cd "${rundir}"

exit ${failed}
