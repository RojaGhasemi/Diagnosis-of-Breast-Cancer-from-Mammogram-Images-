function bw=adaptivethreshold(IM,x,th)

h = ones(x,x) / (x*x);
mIM = imfilter(IM,h);

bw=IM>(mIM*th);
