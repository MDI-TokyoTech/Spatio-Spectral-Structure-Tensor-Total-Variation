function x= myPCG_GSSTV(x,Ysum,M2,F,Gamma,DgDb, DgDbt, beta)
%   temp=beta*Ysum(:)+M2(:)+beta*diffT3(F,dim)-diffT3(Gamma,dim);       
    temp = beta*Ysum(:) + M2(:) + beta.*DgDbt*F - DgDbt*Gamma;     
    [x, ~] = pcg(@(x) Fun(x), temp, 1e-4,1000,[],[],x);   

    function y = Fun(x)
%          y=beta*x+beta*diffT3(diff3(x,dim),dim);
         y = beta*x + beta.*DgDbt*DgDb*x;
    end
end
