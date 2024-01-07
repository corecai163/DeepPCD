%% check chamfer distance error
ours = readmatrix('./result/chd_loss.txt.xyz',FileType='text');

[y,x]=ecdf(ours);
[x,y]