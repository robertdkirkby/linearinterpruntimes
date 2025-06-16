% Test the interpolation
% Point of this is to test runtimes of using interp1() versus manual (index
% and probability) to do linear interpolation of the expected value fn (or
% just any matrix really, where interpolation is of the first dimension of
% the matrix).

% Basic setup
N_a=5000; % tried 500, 5000, 50000, 500000
N_z=31; % tried 9, 90, 900
vfoptions.ngridinterpt=51; % tried 2, 11, 51
n2short=vfoptions.ngridinterpt;
N_aprime=(N_a-1)*n2short+N_a;

% First I do interpolation using cpu, interp1() is fastest
% Next I do interpolation using gpu, interp1() is fastest
% Then I try EV as three dimensions (on gpu), interp1() is fastest

% Conclusion is that linear interpolation is better done using interp1().

% griddedInterpolant() is fastest if you only count the 'evaluate' step,
% but not if you include the 'create' step [because of how I want to use
% them I need to count both steps]

% Matlab also has interpn() and interp1q(), but these do not support GPU as
% of writing.

S=100; % number of runs (report average runtime across them)

%% Do on cpu
% Create fake V
EV=linspace(1,N_a,N_a)'.*linspace(1,2,N_z);

a_grid=cumsum(rand(N_a,1));

aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*vfoptions.ngridinterpt))';

gridinterp_ind=repelem((1:1:N_a)',1+n2short,1);
gridinterp_ind=[gridinterp_ind(1:end-n2short-1);N_a-1]; % lower grid point

gridinterp_prob=[repmat(linspace(1,1/(n2short+1),1+n2short)',N_a-1,1);0]; % prob of lower grid point

% Create interpolated version of EV
timer=zeros(S,2);
for ii=1:S
    tic;
    EVinterp1=interp1(a_grid,EV,aprime_grid);
    timer(ii,1)=toc;

    tic;
    EVinterp2=gridinterp_prob.*EV(gridinterp_ind,:)+(1-gridinterp_prob).*EV(gridinterp_ind+1,:);
    timer(ii,2)=toc;
end
disp('times')
max(timer,[],1)
fprintf('Check they are equal (should be zero): %1.8f \n',max(abs(EVinterp1(:)-EVinterp2(:))))
fprintf('Check they are equal (should be zero): %1.8f \n',max(abs(EVinterp1(:)-EVinterp3(:))))

% There are some numerical rounding error differences at 1e-14:
% EVinterp1-EVinterp2

%% Now move to gpu, which is what we actually care about

% Create fake V
EV=gpuArray(linspace(1,N_a,N_a)'.*linspace(1,2,N_z));
% EV=gpuArray(cumsum(cumsum(rand(N_a,N_z),2),1)); % alternative [make sure it wasn't anything to do with me using integers; it wasn't]
% Note: only tests out EV that is increasing in each of (a,z), as this is standard in my applications

a_grid=gpuArray(cumsum(rand(N_a,1)));

aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*vfoptions.ngridinterpt))';

gridinterp_ind=repelem((1:1:N_a)',1+n2short,1);
gridinterp_ind=gpuArray([gridinterp_ind(1:end-n2short-1);N_a-1]); % lower grid point
gridinterp_indplus1=gridinterp_ind+1;

gridinterp_ind2=gridinterp_ind+N_a*(0:1:N_z-1); % make it a full matrix index
gridinterp_ind2plus1=gridinterp_ind+1+N_a*(0:1:N_z-1); % make it a full matrix index

gridinterp_prob=gpuArray([repmat(linspace(1,1/(n2short+1),1+n2short)',N_a-1,1);0]); % prob of lower grid point
gridinterp_probB=1-gridinterp_prob; % prob of upper grid point


% Create interpolated version of EV
timer=zeros(S,5);
for ii=1:S
    tic;
    EVinterp1=interp1(a_grid,EV,aprime_grid);
    timer(ii,1)=toc;

    tic;
    EVinterp2=gridinterp_prob.*EV(gridinterp_ind,:)+gridinterp_probB.*EV(gridinterp_indplus1,:);
    timer(ii,2)=toc;

    % If I do the whole 2D index, that makes it faster
    tic;
    EVinterp3=gridinterp_prob.*EV(gridinterp_ind2)+gridinterp_probB.*EV(gridinterp_ind2plus1);
    timer(ii,3)=toc;

    % Tried to see if repelem() is faster than indexing, but it is slower
    tic;
    EVrepeated=repelem(EV,1+n2short,1);
    EVinterp4=gridinterp_prob.*EVrepeated(1:end-n2short,:)+gridinterp_probB.*EVrepeated(n2short+1:end,:);
    timer(ii,4)=toc;

    % griddedInterpolant
    tic;
    F=griddedInterpolant(a_grid,EV);
    timer(ii,6)=toc;
    tic;
    EVinterp5=F(aprime_grid);
    timer(ii,7)=toc;
    timer(ii,5)=timer(ii,6)+timer(ii,7);
end
disp('times')
max(timer,[],1)
fprintf('Check they are equal (should be zero): %1.8f \n',max(abs(EVinterp1(:)-EVinterp2(:))))
fprintf('Check they are equal (should be zero): %1.8f \n',max(abs(EVinterp1(:)-EVinterp3(:))))
fprintf('Check they are equal (should be zero): %1.8f \n',max(abs(EVinterp2(:)-EVinterp3(:))))
fprintf('Check they are equal (should be zero): %1.8f \n',max(abs(EVinterp1(:)-EVinterp4(:))))
fprintf('Check they are equal (should be zero): %1.8f \n',max(abs(EVinterp3(:)-EVinterp4(:))))
fprintf('Check they are equal (should be zero): %1.8f \n',max(abs(EVinterp1(:)-EVinterp5(:))))
fprintf('Check they are equal (should be zero): %1.8f \n',max(abs(EVinterp3(:)-EVinterp5(:))))

% First and third are essentially same runtime when N_a=5000, but first is
% marginally faster when N_a=50000. With N_a=500000, the runtimes seem
% unstable and occasionally third is fastest, occasionally first is
% fastest; did a bunch more, mostly first is fastest.

%% Does adding a third dimension make any difference to this?

N_a2=21;

% Create fake V
EV1=repelem(linspace(1,N_a,N_a)',1,N_a2,N_z);
EV2=repelem(linspace(1,N_a2,N_a2),N_a,1,N_z);
EV3=repelem(shiftdim(linspace(1,2,N_z),-1),N_a,N_a2,1);
EV=EV1.*EV2.*EV3;

a_grid=gpuArray(cumsum(rand(N_a,1)));

aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*vfoptions.ngridinterpt))';

gridinterp_ind=repelem((1:1:N_a)',1+n2short,1);
gridinterp_ind=gpuArray([gridinterp_ind(1:end-n2short-1);N_a-1]); % lower grid point
gridinterp_indplus1=gridinterp_ind+1;

gridinterp_ind2=gridinterp_ind+N_a*(0:1:N_a2-1)+N_a*N_a2*shiftdim((0:1:N_z-1),-1); % make it a full matrix index
gridinterp_ind2plus1=gridinterp_ind+1+N_a*(0:1:N_a2-1)+N_a*N_a2*shiftdim((0:1:N_z-1),-1); % make it a full matrix index

gridinterp_prob=gpuArray([repmat(linspace(1,1/(n2short+1),1+n2short)',N_a-1,1);0]); % prob of lower grid point

% Create interpolated version of EV
timer=zeros(S,2);
for ii=1:S
    tic;
    EVinterp1=interp1(a_grid,EV,aprime_grid);
    timer(ii,1)=toc;

    tic;
    EVinterp2=gridinterp_prob.*EV(gridinterp_ind,:,:)+(1-gridinterp_prob).*EV(gridinterp_indplus1,:,:);
    timer(ii,2)=toc;

    tic;
    EVinterp3=gridinterp_prob.*EV(gridinterp_ind2)+(1-gridinterp_prob).*EV(gridinterp_ind2plus1);
    timer(ii,3)=toc;
end
disp('times')
max(timer,[],1)
fprintf('Check they are equal (should be zero): %1.8f \n',max(abs(EVinterp1(:)-EVinterp2(:))))
fprintf('Check they are equal (should be zero): %1.8f \n',max(abs(EVinterp1(:)-EVinterp3(:))))
fprintf('Check they are equal (should be zero): %1.8f \n',max(abs(EVinterp2(:)-EVinterp3(:))))

% interp1() was the clear winner, in fact this widens the gap

% Here the second is actually faster than the third.
% The first is way faster.

