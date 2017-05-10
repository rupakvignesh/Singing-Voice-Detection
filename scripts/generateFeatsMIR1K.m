%generateFeats MIR1K
clear all; close all; clc;
winSize = 0.040;
hopSize = 0.020;
sr = 16000;
fftSize = 2^(ceil(log(winSize*sr)/log(2)));

ipdir = '/Users/RupakVignesh/Desktop/spring17/7100/Data/MIR-1K/Train/';
%opdir = '/Users/RupakVignesh/Desktop/spring17/7100/Data/test_feats/';
gt_labels = '/Users/RupakVignesh/Desktop/spring17/7100/Data/MIR-1K/vocal-nonvocalLabel/';
list = dir(strcat(ipdir,'*.wav'));


features = [];
for i=1:length(list)
    [y,fs] = audioread(strcat(ipdir,list(i).name));
    y = y(:,2); % Downmix
    y = resample(y,sr,fs); % Downsample
    [~, filename,~] = fileparts(list(i).name);
    fileID = fopen(strcat(gt_labels,filename,'.vocal'),'r');
    formatSpec = '%f';
    GT = fscanf(fileID,formatSpec);
    fclose(fileID);
     %Y = spectrogram(y,round(winSize*sr),round(sr*winSize)-round(sr*hopSize),fftSize,sr);
    Y = computeGroupDelaySpectra(y,sr,winSize,hopSize,0);
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
    
    
    if length(GT)>size(Y,2)
        GT = GT(1:size(Y,2));
    elseif length(GT)<size(Y,2)
        GT(length(GT)+1:size(Y,2)) = 0;
    end
        
    features = [features; [mfccs', SpecCentroid', SpecFlux', SpecCrest' ...
        SpecFlatness', SpecRolloff', SpecSlope', SpecSpread', ...
        SpecSkewness', GT]]; %d=47
    
    %features = [features; Y', GT];
    features(isnan(features)) = exp(-200);
end
