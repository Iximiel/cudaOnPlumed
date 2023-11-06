#! /usr/bin/env bash

source "$PLUMED_ROOT"/src/config/compile_options.sh

#pendantic adds a unuseful FOR EACH line with
#"" warning: style of line directive is a GCC extension"
compile=${compile//-pedantic/}
if [[ ${SILENT_CUDA_COMPILATION} ]]; then
  #echo "disabled warning"
  compile=${compile//-Wall/}
  #-w suppress the warnings
  compile=${compile/-c /-w -c }
fi

for opt in -W -pedantic -f; do
  compile=${compile//${opt}/-Xcompiler ${opt}}
done

link_command=$link_uninstalled

if [[ -z ${link_command:+x} ]]; then
  link_command=$link_installed
fi

link_command=${link_command#*-shared}
link_command=${link_command/-rdynamic/-Xcompiler -rdynamic}
link_command=${link_command/-Wl,/-Xlinker }
#link_command=${link_command/-fopenmp/-Xcompiler -fopenmp}
for opt in -f; do
  link_command=${link_command//${opt}/-Xcompiler ${opt}}
done

compile=${compile// -o/}
link_command=${link_command// -o/}

cat <<EOF >make.inc
compile := $compile
link    := $link_command
EOF
