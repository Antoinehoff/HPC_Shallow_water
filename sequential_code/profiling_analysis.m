%% profiling analysis
% Import data from text file.
filename = '/home/antoine/EPFL/HPC/Project_Shallow_water/sequential_code/manual_profiling.txt';
delimiter = ',';
formatSpec = '%f%f%f%f%f%f%f%f%f%[^\n\r]';
fileID = fopen(filename,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'TextType', 'string',  'ReturnOnError', false);
fclose(fileID);
manualprofiling = [dataArray{1:end-1}];
clearvars filename delimiter formatSpec fileID dataArray ans;

% Process data
results = [];
names = ["Var. init.", "Init. State Load.", "dt updating",...
         "Copy Temp. Var.", "Enf. BC", "Time step perf.",...
         "Impose tol.", "Save output", "Free memspace"];
sum_ = sum(manualprofiling');
for i=1:9
    results(i,1) = mean(manualprofiling(:,i));
    results(i,2) = std(manualprofiling(:,i));
    results(i,3) = mean(manualprofiling(:,i)'./sum_)*100.0;
    results(i,4) = std(manualprofiling(:,i)'./sum_)*100.0;
end