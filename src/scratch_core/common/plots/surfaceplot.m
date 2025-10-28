function Iout = GetSurfacePlot(data_in, varargin)

%
% surface_plot = GetSurfacePlot(data_in)
% surface_plot = GetSurfacePlot(data_in, extra_light_source)
% surface_plot = GetSurfacePlot(data_in, extra_light_source, light_angles)
% surface_plot = GetSurfacePlot(data_in, extra_light_source, light_angles, mask)
% surface_plot = GetSurfacePlot(data_in, extra_light_source, light_angles, mask, fill_display)

% GetSurfacePlot is used to generate a surface rendering of the input image
% DATA_IN. The viewer position is along the z-axis and light sources are
% placed by default at two directions.
% Optionally, an array with LIGHT_ANGLES can be provided.
% If a MASK is provided, the image is masked prior to surface determination
% If FILL_DISPLAY is set to TRUE (1), then the bounding box of the masked
% data will be determined and returned instead of the entire image. This
% results in the image to be shown filling the display.
%
% Input:
%   data_in                 Scratch data structure or image double array
%   extra_light_source      Scalar indicating whether an additional lights source must be turned on (default = 0)
%   light_angles            n x 2 array with n light sources at angles [az el]
%   mask                    Array of same size as data_in.depth_data, with
%                           foreground > 0 and background = 0
%   fill_display            Scalar indicating whether the masked area should be
%                           stretched to fill the display, to improve the
%                           presentation if selected areas are small (e.g. aperture
%                           shear) (default = 0)
% Output:
%   Iout                    Image of rendered surface
%
% Information:
%   Azimuth (az):       Rotation around z-axis in degrees
%   Elevation (el):     Elevation w.r.t. the x-y-plane in degrees
%   Default light angles (works best for unprocessed measurement data):
%   az =  90, el = 45:  Light from the right, at an elevation of 45 deg
%   az = 180, el = 45:  Light from the bottom, at an elevation of 45 deg
%

% User settings
doplot = 0;                     % 0/1 = no/yes (only for development purposes)
light_angles = [90,45; 180,45]; % azimuth, elevation of each light source [deg]

% Retrieve data needed, forget the rest
depthdata = data_in.depth_data;
xdim      = data_in.xdim;
ydim      = data_in.ydim;
clear data_in

% Plot depthdata
if doplot
    figure(); imagesc(depthdata); axis equal on; colormap gray
end

% Turn an extra light source on if requested
if nargin > 1 && varargin{1}
    light_angles = [90,45; 180,45; 270,45];
end

% If light sources are provided, these replace the defaults
if nargin > 2 && ~isempty(varargin{2})
    light_angles = varargin{2};
end

% If a mask is provided, mask the image before generating the surface
if nargin > 3 && ~isempty(varargin{3})
    mask = varargin{3};

    % If the data in GetPreviewImageForCropping got reduced as a result of a very small
    % pixelsize, the data sent to GetSurfacePlot has not the same size as the mask anymore.
    % Therefore the mask has to be reduced in size.
    if any( size(mask) ~= size(depthdata) )
        %error('GetsurfacePlot: Mask and depth data dimensions mismatch!');
        mask = imresize(mask, size(depthdata), 'nearest');
    end

    % Mask background data with NaN
    depthdata(~mask) = nan;
else
                mask = ones(size(depthdata)); % everything is foreground, nothing to mask
end

% Set fill_display
if nargin > 4
    fill_display = varargin{4};
else
    fill_display = 0;
end

% If the fill display option is set, remove the boundaries
if fill_display
    bbox      = DetermineBoundingBox(mask);
    depthdata = depthdata( bbox(2,1):bbox(2,2), bbox(1,1):bbox(1,2));
end


% Create vector pointing towards observer
OBS = getv(  0,90);                 % azimuth 0 deg , elevation 90 deg

% Create surface normals
% Surface: z=ax+by+c ==> -ax-by+z=c ==> surface normal (-a,-b,1)
% Normalization: sqrt(a*a+b*b+1*1)

hx = diff(depthdata,1,1)/xdim;      % slope x-direction
hy = diff(depthdata,1,2)/ydim;      % slope y-direction

hx(end+1,:) = nan;                  % to make dimensions equal
hy(:,end+1) = nan;                  % to make dimensions equal

norm = sqrt( hx.*hx + hy.*hy + 1);  % normalization
n1 = -hx./norm;                     % surface normal 1st dimension
n2 = -hy./norm;                     % surface normal 2nd dimension
n3 =   1./norm;                     % surface normal 3rd dimension

% Calculate intensity of surface for each light source
nLS  = size(light_angles,1) ;                           % Number of light sources
Iout = nan(size(depthdata,1), size(depthdata,2), nLS);  % Allocate memory

for i=1:nLS
    LS = getv( light_angles(i,1), light_angles(i,2) );  % Create vector pointing towards light source
    Iout(:,:,i) = calcsurf(LS,OBS,n1,n2,n3);            % Calc intensity from surface due to light source
    if doplot
        figure(); imshow(Iout(:,:,i)); axis equal off; colormap gray;
    end
end

% Calculate total intensity of surface
Iout = sum(Iout,3);         % add up result of all light sources

% Normalize between [0,1] (for plotting purposes)
Imin = min(Iout(:));
Imax = max(Iout(:));
Iout = (Iout-Imin)/(Imax-Imin);

% Add ambient component and scale from [0,1]->[0,255]
famb = 25;
Iout = famb + (255-famb)*Iout;

end

%--------------------------------------------
function v = getv(az,el)
v = [-cosd(az)*cosd(el); sind(az)*cosd(el); sind(el)];
end

%--------------------------------------------
function Iout = calcsurf(LS,OBS,n1,n2,n3)

% PREPARATIONS
h = LS+OBS;         % Vector corresponding to normal of surface that produce max specular reflection
h = h./sqrt(h'*h);  % Normalize vector

% DIFFUSE COMPONENT
Idiffuse = LS(1)*n1 + LS(2)*n2 + LS(3)*n3;  % inner product light source and surface normal
Idiffuse(Idiffuse<0)=0;                     % if <0  no light falls upon facet, so nothing to diffuse

% SPECULAR COMPONENT
Ispec = h(1)*n1 + h(2)*n2 + h(3)*n3;        % inner product halfway vector and surface normal

Ispec(Ispec<0)=0;             % if <0  no light, so nothing to reflect
Ispec = cos(2*acos(Ispec));   % double the angle (is this an approximation?)
Ispec(Ispec<0)=0;             % if <0  no light

% f=4;                        % phong factor (typically between 1 and 10)
% Ispec = Ispec.^f;           % phong
Ispec = Ispec.*Ispec.*Ispec.*Ispec;    % phong with f=4

% DIFFUSE + SPECULAR
fspec = 1;
Iout  = (Idiffuse + fspec*Ispec)/(1+fspec);

end

%--------------------------------------------
function bounding_box = DetermineBoundingBox(mask)

x_sum = sum(mask,1);
y_sum = sum(mask,2);

start_x = 1;
end_x   = length(x_sum);
start_y = 1;
end_y   = length(y_sum);

for x_cnt = 1:length(x_sum)
    if x_sum(x_cnt)
        start_x = x_cnt;
        break;
    end
end

for x_cnt = length(x_sum):-1:1
    if x_sum(x_cnt)
        end_x = x_cnt;
        break;
    end
end

for y_cnt = 1:length(y_sum)
    if y_sum(y_cnt)
        start_y = y_cnt;
        break;
    end
end

for y_cnt = length(y_sum):-1:1
    if y_sum(y_cnt)
        end_y = y_cnt;
        break;
    end
end

bounding_box = [start_x end_x; start_y end_y];

end
