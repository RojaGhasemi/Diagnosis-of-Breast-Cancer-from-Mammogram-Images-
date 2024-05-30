clear all;
clc;
numberoffeature=30;
%% read dataset images
bei=ls('dataset\b');
bei=bei(3:end,:);
can=ls('dataset\c');
can=can(3:end,:);
nom=ls('dataset\n');
nom=nom(3:end,:);

for i=1:size(nom,1)
    nom2(i,:)=['dataset\n\' nom(i,:)];
    nom=nom2;
end


for i=1:size(can,1)
    can2(i,:)=['dataset\c\' can(i,:)];
    can=can2;
end


for i=1:size(bei,1)
    bei2(i,:)=['dataset\b\' bei(i,:)];
    bei=bei2;
end

%%  read all images
se = strel('disk',5);
tic
traindata=[nom;can;bei];
% define gabor wavelets
wavelength = [2 8 16];
orientation =0:45:180;
g = gabor(wavelength,orientation);
for j=1:size(traindata,1)    
    tic
    j/size(traindata,1)*100
    [images map]=imread(traindata(j,:), 'gif','Frames','all');
    outpict=rgb2gray(ind2rgb8(images,map));        
    %step one resize images
    image=imresize(outpict,[680 397]);    
    %step two average filter images
    h = ones(9,9) / 81;
    I2 = imfilter(image,h);
    % step three adaptive thresh and save that ones equal too 1
    mask=adaptivethreshold(I2,3,0.9);    
    I2(find(mask==0))=0;
    % step 4 avearaging and Adaptive thresholding again with 39*39
    h = ones(39,39) / (39*39);
    I3 = imfilter(I2,h);
    mask=adaptivethreshold(I3,1,0.9964);        
    I2(find(mask==0))=0;
    % step 5 image aspect ratio greater than 1.7 is removed
    I=I2>240;
    OO=find(I==1);
    [Oj,Oi]=ind2sub([680,397],find(I==1));
    Oi=Oi./Oj;
    Oi2=Oj./Oi;
    for k=1:size(Oi,1)
        if ((Oi(k)>=1.7)|(Oi2(k)>=1.7))
            I2(OO(k))=0;
        end
    end
    % step 6 top hat and botom hat transform for image enhancement
    I2 = imsubtract(imadd(I2,imtophat(I2,se)), imbothat(I2,se));    
    % step 7 resize image to a low size image for gabor filtering
    I2=imresize(I2,[64 64]);
    % step 8 apply gaboor filter 	
    I2=im2double(I2);
	outMag = imgaborfilt(I2,g);       
    outMag=reshape(outMag,numel(outMag),1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    allimages(j,:)=outMag; 
    toc
end
labels(1:size(traindata))=1;
labels(size(nom2,1)+1:size(traindata))=2;
labels(size(nom2,1)+size(can,1)+1:size(traindata))=3;
toc
disp('preprocessing and feature extraction is completed');
% step 9 aplly LSDA for dimention reduction

[eigvector, eigvalue]=LSDA(labels,allimages);
%[eigvector, eigvalue]=LDA(labels,allimages);

allimages=allimages*eigvector;
disp('dimention reduction is completed');
% step 10 select top best feature
allimages=allimages(:,1:numberoffeature);
% 10 fold cross validataion get indices for kfold of data
indices = crossvalind('Kfold',traindata,10);

errorknn=0;
errorSVMRBF=0;
errorSVMPoly3=0;
%% traininig and testing classification models
for i=1:10
    %% divide dataset to test and train for Kfold corss validation
    tic;
    testbe = (indices == i); 
    testdata=allimages(testbe,:);
    testlabel=labels(testbe);
    testbe = ~testbe;
    traindata=allimages(testbe,:);            
    trainlabel=labels(testbe);
    class = knnclassify(testdata, traindata, trainlabel);
    errorknn=errorknn+sum(class==testlabel')/size(testlabel,2);
    t = templateSVM('Standardize',1,'KernelFunction','gaussian');
    t2 = templateSVM('Standardize',1,'KernelFunction','polynomial');
    svmRBF = fitcecoc(traindata,trainlabel,'Learners',t);    
    svmPoly3 = fitcecoc(traindata,trainlabel,'Learners',t2);
    class = predict(svmRBF,testdata);
    errorSVMRBF=errorSVMRBF+sum(class==testlabel')/size(testlabel,2);
    class = predict(svmPoly3,testdata);
    errorSVMPoly3=errorSVMPoly3+sum(class==testlabel')/size(testlabel,2);
end
disp('knn error is: ');
errorknn=errorknn/10
errorSVMRBF=errorSVMRBF/10
errorSVMPoly3=errorSVMPoly3/10