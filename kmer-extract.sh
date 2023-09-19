tail -n +0 ./name.txt | 
	while read name; do
                kmc -k11 -m24 /path/to/${name}.r1.fq.gz ${name}_out1 ./
                kmc -k11 -m24 /path/to/${name}.r2.fq.gz ${name}_out2 ./
                kmc_tools simple ${name}_out1 ${name}_out2 intersect ${name}_inter
                kmc_dump ${name}_inter ./k11-${name}.txt
	done
