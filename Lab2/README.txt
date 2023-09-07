The code compiles fine with the flags mentioned in the report
for the csl-artemis/csl-venus. In case of testing on another machine, try:
1. Changing icx to icc or,
2. Changing the -fast flag to -O4 and add an -mfma flag to OPTFLAGS
