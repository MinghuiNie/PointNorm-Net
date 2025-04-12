function [W normals ] = compute_normal_PCA( Pts, knnSize )
if nargin == 1
    knnSize = 500;
end

nSamples = length(Pts);
normals = zeros(nSamples ,3);
W = zeros(nSamples ,1);
kdtree = kdtree_build(Pts);

for  i = 1 : nSamples

    %neis = kdtree_ball_query(kdtree, Pts(i,:), 0.1);
    neis = kdtree_k_nearest_neighbors(kdtree, Pts(i,:), knnSize);
    neiPoints = Pts(neis,:) ;
    
    if length(neis) > knnSize
        neiPoints = Pts(neis(1:500),:) ;
    else
        neiPoints = Pts(neis,:) ;
    end
    
    neiPoints = neiPoints - ones(size(neiPoints , 1), 1)*mean(neiPoints) ;
    dis = max(sqrt(sum(neiPoints.^2, 2))) ;
    neiPoints = neiPoints ./ dis;
    
    %% PCA
    [s,v] = computePCA(neiPoints);
    s = diag(s) ;
    W(i) = s(1)/sum(s) ;
    normals(i,:) = v(:,1)';

end
kdtree_delete(kdtree);
end


