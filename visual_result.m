%% show result
close all
%load('initial_result.mat')
index =1;
o1 = incomplete;
p1 = pred;
g1 = gtcomplete;
%g1 = patch_complete(10241:end,:);

g1 = squeeze(g1);
o1 = squeeze(o1);
p1 = squeeze(p1);

gt_pcd = pointCloud(g1);
or_pcd = pointCloud(o1);
pt_pcd = pointCloud(p1);
% gt_pcd = pointCloud(g1,'Color',gc1);
% or_pcd = pointCloud(o1,'Color',oc1);
% pt_pcd = pointCloud(p1,'Color',pc1);
%init_pcd = pointCloud(ini_pt(1:2048,:));

figure
pcshow(or_pcd,'MarkerSize',53)
set(gcf,'color','w');
set(gca,'color','w');

figure
pcshow(pt_pcd,'MarkerSize',53)
set(gcf,'color','w');
set(gca,'color','w');

% figure
% pcshow(init_pcd,'MarkerSize',53)
% set(gcf,'color','w');
% set(gca,'color','w');

figure
pcshow(gt_pcd,'MarkerSize',53)
set(gcf,'color','w');
set(gca,'color','w');

% x = linspace(0,1,16);
% [X,Y,Z] = meshgrid(x,x,x);
% x = reshape(X,[],1);
% y = reshape(Y,[],1);
% z = reshape(Z,[],1);
% points = [x,y,z];
% blank = pointCloud([points;points]);
% pcshow(blank, 'MarkerSize',13)
% set(gcf,'color','w');
% set(gca,'color','w');
