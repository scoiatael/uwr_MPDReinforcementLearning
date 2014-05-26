#!/bin/zsh

DATA=$1
MINX=$2
MAXX=$3

function add_col {
  local result_name=$1
  echo ${(P)result_name}" '$DATA' u (column(0)):$2 s csplines title '$3'"
}

COM="set output '$1.png'\n set terminal png\n set yrange [-$MINX:$MAXX] \nplot"
COM=$(add_col COM 9 "(1,1)")
COM=$COM","
COM=$(add_col COM 11 "(3,1)")
COM=$COM","
COM=$(add_col COM 3 "(3,3)")
COM=$COM","
COM=$(add_col COM 12 "(4,1)")
COM=$COM","
COM=$(add_col COM 4 "(4,3)")
COM=$COM","
COM=$(add_col COM 2 "(2,3)")
echo $COM

