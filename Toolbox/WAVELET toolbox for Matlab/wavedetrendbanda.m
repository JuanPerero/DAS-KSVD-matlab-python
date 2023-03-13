function [salida windowef]=wavedetrendbanda(x,win,step,level,wave,number)

xx=x;

wsize=length(win);
n=length(xx);
p=floor((n-wsize)/step);
r=p*step;

[rh,rg]=wavefilters(number,wave);

ll=maxlevel(zeros(1,wsize));
if (ll<level),
    disp('Error, nivel de descomposicion imposible para esa longitud de ventana');
    return;
end
g=floor(wsize/step);
salida=zeros(size(xx));
windowef=zeros(size(salida));

for j=1:p,
    ini=(j-1)*step+1;
    fin=ini+wsize-1;
    pedazo = xx(ini:fin).* win';
   
    %dwt
    r=1;
    [c,l]=dwtdecomp(pedazo,rh,rg,level);
    
    c(1:sum(l(1:r)))=0;  %%filtro pasaalto
    c(l(end-3)+1:end)=0; %%filtro pasabajo 
    
    salida(ini:fin)=salida(ini:fin)+dwtreconst(c,l,rh,rg)';
    windowef(ini:fin)=windowef(ini:fin)+win';
    
end
% if (r ~= n),
%     ini=fin+1;
%     pedazo=zeros(1,wsize);
%     nl=n-ini+1;
%     fin=n;
%     pedazo(1:nl)=xx(ini:fin);
%     pedazo=pedazo.*win;
%     [c,l]=dwtdecomp(pedazo,rh,rg,level);
%     c(1:l(1))=0;
%  %   c(17:end)=0;
%     dd=dwtreconst(c,l,rh,rg);
%     salida(ini:fin)=salida(ini:fin)+dd(1:nl)';
%     windowef(ini:fin)=windowef(ini:fin)+win(1:nl)';
% end

%plot(windowef)

%salida=salida(wsize+1:end-wsize)/g;
salida=salida./windowef;
