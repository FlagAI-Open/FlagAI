fil=$1
for i in {17,20};do
	scp $fil 192.168.$i.2:/data2/gitee/flagai-internal/examples/aquila/
done
