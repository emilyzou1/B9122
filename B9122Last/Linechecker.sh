#!/bin/bash


count=$(cat $1 | wc -l)


function numlines(){
	
	if [[ $count -eq 0 ]]; then
		echo file is empty
	elif [[ $count -lt 10 ]]; then
		echo file has less than or equal to 10 lines
	else
		echo file has more than 10 lines

	fi

}
#output is done
echo $(numlines $count)