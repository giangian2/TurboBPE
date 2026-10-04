# GDB setup for stepping through assembly
set disassembly-flavor intel
set pagination off
set confirm off

# Show registers + next instructions every time execution stops
layout split
layout regs
focus cmd

break main
run
