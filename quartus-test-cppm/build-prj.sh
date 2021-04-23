#!/bin/bash

basedir=quartus_prj
parallel=1
ctest=0


function print_usage {
   echo "Usage: `basename $0` [OPTION]"
   echo ""
   echo "Builds Quartus HLS projects found in the current directory."
   echo ""
   echo "Options are:"
   echo "   -d DIR"
   echo "      Base directory of projects to build. Defaults to 'quartus_prj'."
   echo "   -n"
   echo "      Create new project (reset any existing)."
   echo "   -h"
   echo "      Prints this help message."
}

function run_quartus {
   dir=$1
   echo "Building project in ${dir}"
   cd ${dir}
   cmd="make myproject-fpga"
   eval ${cmd}
   if [ $? -eq 1 ]; then
      touch BUILD_FAILED
   fi
   cd ..
   return ${failed}
}

function run_ctest {
   dir=$1
   echo "Building ctest-project in ${dir}"
   cd ${dir}
   cmd="make myproject-ctest"
   eval ${cmd}
   if [ $? -eq 1 ]; then
      touch BUILD_FAILED
   fi
   cd ..
   return ${failed}
}
function run_simulation {
   dir=$1
   echo "Running sim in ${dir}"
   cd ${dir}
   cmd="make myproject-fpga"
   eval ${cmd}
   if [ $? -eq 1 ]; then
      touch SIM_FAILED
   fi
   cd ..
   return ${failed}
}

while getopts ":d:nhic" opt; do
   case "$opt" in
   d) basedir=$OPTARG
      ;;
   n) reset="reset=1"
      ;;
   h)
      print_usage
      exit
      ;;
   c)
      ctest=1
      ;;
   :)
      echo "Option -$OPTARG requires an argument."
      exit 1
      ;;
   esac
done

if [ ! -d "${basedir}" ]; then
   echo "Specified directory '${basedir}' does not exist."
   exit 1
fi

#rundir=`pwd`

cd "${basedir}"

for project in $(ls -d */) ; do
  echo "$project"
  if [ $ctest -eq 1 ]; then
    run_ctest "${project}"
    exit ${failed}
  fi
  run_quartus "${project}"
  run_simulation "${project}"
done
exit ${failed}
