#!/usr/bin/bash
set -e
set -x

# odin build . -debug -out:prog.bin
odin test . -debug -out:prog.bin 
# odin test . -o:speed -out:prog.bin 
# ./prog.bin
