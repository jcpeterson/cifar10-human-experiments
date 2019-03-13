#!/bin/bash

declare -a jobs=()

if [ -z "$1" ] ; then
    echo "Minimum Job Number argument is required.  Run as '$0 jobnum jobnum2'"
    exit 1
fi

if [ -z "$2" ] ; then
    echo "Maximum Job Number argument is required.  Run as '$0 jobnum jobnum2'"
    exit 1
fi


minjobnum="$1"
maxjobnum="$2"
myself="$(id -u -n)"

for j in $(squeue --user="$myself" --noheader --format='%i') ; do
  if [ "$j" -ge "$minjobnum" ]; then
    if [ "$j" -le "$maxjobnum" ]; then
    jobs+=($j)
    fi	
  fi
done

echo "${jobs[@]}"
scancel "${jobs[@]}"
