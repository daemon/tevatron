awk 'BEGIN{OFS="\t"} {if ($1!=last){rank=1; last=$1}else{rank++} print $1,"Q0",$2,rank,$3,"adder"}'  $1/rankings.pkl > $2
