clc ; clear ; close all

point_filename = '..\testData\pclouds' ;
save_filename = '..\testData\pclouds_out' ;

filesPre = dir(fullfile(point_filename,'*.xyz'));
knnSize = [16 32 64 128 256 350];
numfile = length(filesPre) ;
RMS_tao = zeros(numfile , 6) ; RMS = zeros(numfile , 6) ;
W_mean = zeros(numfile , 6) ;
for i = 1 :numfile
    i
    cur_pointfile = [point_filename '\' filesPre(i).name] ;
    
    pointfid = fopen(cur_pointfile);
    points = fscanf(pointfid, '%f %f %f', [3 inf]);
    fclose(pointfid);
    points = points';
    
    features = zeros(size(points, 1) , 18) ;
    W = zeros(size(points, 1) , 6) ;
    for ii = 1 : length(knnSize)
        [W(:,ii) features(:, (ii-1)*3+1:ii*3)] = compute_normal_PCA(points ,  knnSize(ii)) ;
        %[~ , RMS(i,ii)] = evaluate( trueNormals, features(:, (ii-1)*3+1:ii*3)) ;
    end
    W_mean(i,:) = mean(W) ;
    save_file = [ save_filename '\' filesPre(i).name(1:end-4) '.features'] ;
    fid=fopen(save_file,'wt');
    fprintf(fid,'%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n',features');
    fclose(fid);
    
    save_file = [ save_filename '\' filesPre(i).name(1:end-4) '.weights'] ;
    fid=fopen(save_file,'wt');
    fprintf(fid,'%f %f %f %f %f %f\n',W');
    fclose(fid);
    
end
