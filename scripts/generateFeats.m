clear all; close all; clc;
winSize = 0.032;
hopSize = 0.016;
sr = 16000;
fftSize = 2^(ceil(log(winSize*sr)/log(2)));

ipdir = '/Users/RupakVignesh/Desktop/spring17/7100/Data/valid/';
%opdir = '/Users/RupakVignesh/Desktop/spring17/7100/Data/test_feats/';
gt_labels = '/Users/RupakVignesh/Desktop/spring17/7100/Data/gt_labels/';
list = [dir(strcat(ipdir,'*.ogg')); dir(strcat(ipdir,'*.mp3'))];


features = [];
for i=1:length(list)
    [y,fs] = audioread(strcat(ipdir,list(i).name));
    y = mean(y,2); % Downmix
    y = resample(y,sr,fs); % Downsample
    [~, filename,~] = fileparts(list(i).name);
    fileID = fopen(strcat(gt_labels,filename,'.lab'),'r');
    formatSpec = '%f';
    GT = fscanf(fileID,formatSpec);
    
     %Y = spectrogram(y,round(winSize*sr),round(sr*winSize)-round(sr*hopSize),fftSize,sr);
    Y = modgd(y, fftSize, round(winSize*sr), round(sr*hopSize), sr);
     Y = abs(Y);
     
     M = melfcc(y,sr,'wintime',winSize,'hoptime',hopSize);
     Mdel = deltas(M);
     Mdeldel = deltas(deltas(M,5),5);
     mfccs = [M; Mdel; Mdeldel]; 
     
     
     SpecCentroid = FeatureSpectralCentroid(Y,sr);
     SpecFlux = FeatureSpectralFlux(Y,sr);
     SpecCrest = FeatureSpectralCrestFactor(Y,sr);
     SpecFlatness = FeatureSpectralFlatness(Y,sr);
     SpecRolloff = FeatureSpectralRolloff(Y,sr,0.85);
     SpecSlope = FeatureSpectralSlope(Y,sr);
     SpecSpread = FeatureSpectralSpread(Y,sr);
     SpecSkewness = FeatureSpectralSkewness(Y,sr);
     %SpecKurtosis = FeatureSpectralKurtosis(Y,sr);
     %Zcr = FeatureTimeZeroCrossingRate(y,winSize,hopSize,sr); 
    
    
    if length(GT)>length(Y)
        GT = GT(1:length(Y));
    elseif length(GT)<length(Y)
        GT(length(GT)+1:length(Y)) = 0;
    end
        
    features = [features; [mfccs', SpecCentroid', SpecFlux', SpecCrest' ...
        SpecFlatness', SpecRolloff', SpecSlope', SpecSpread', ...
        SpecSkewness', GT]]; %d=47
    
    %features = [features; Y', GT];
    features(isnan(features)) = exp(-200);
end
