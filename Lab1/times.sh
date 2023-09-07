#/bin/bash

cd O0

declare -a versions=("01.original" "02.loop_interchange_conv_2D" "03.loop_interchange_sobel" "04.loop_interchange_conv2D_sobel" "05.loop_unrolling_conv2d" "06.loop_fusion"\ 
"07.function_inline" "08.strength_reduction" "09.loop_code_motion" "10.subexpression_elimination" "11.single_loop" \
"12.loop_unrolling_2" "13.loop_unrolling_4" "14.loop_unrolling_8" "15.compiler_help" "16.sqrt_lut")

for version in ${versions[@]}
do
	cd ${version}
	cp ../../makefile .
	cp ../../input.grey .
	cp ../../golden.grey .

	sed -i s/"fast"/"O0"/ makefile
	make clean
	make

	rm -rf temp.out times.out

	for i in {1..25}
	do
		./sobel >> temp.out
	done
	
	echo -e -n "$version," >> ../../times.csv
	
	grep "Total time =.*seconds" temp.out >> times.out

	sed -i s/"Total time =[\t ]*"// times.out
	sed -i s/"seconds"// times.out

	while read -r line
	do 
		echo -n -e "$line," >> ../../times.csv;
	done < times.out

	echo "" >> ../../times.csv
	rm -rf temp.out times.out
	make clean
	rm makefile input.grey golden.grey
	cd ../
done

cd ../fast

for version in ${versions[@]}
do
	cd ${version}
	cp ../../makefile .
	cp ../../input.grey .
	cp ../../golden.grey .

	sed -i s/"O0"/"fast"/ makefile
	make clean
	make

	rm -rf temp.out times.out

	for i in {1..25}
	do
		./sobel >> temp.out
	done
	
	echo -e -n "$version," >> ../../times_fast.csv
	
	grep "Total time =.*seconds" temp.out >> times.out

	sed -i s/"Total time =[\t ]*"// times.out
	sed -i s/"seconds"// times.out

	while read -r line
	do 
		echo -n -e "$line," >> ../../times_fast.csv;
	done < times.out

	echo "" >> ../../times_fast.csv
	rm -rf temp.out times.out
	make clean
	rm makefile input.grey golden.grey
	cd ../
done
