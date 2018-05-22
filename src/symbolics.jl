using SymPy

@syms P_ij,P_im1j,P_ip1j,P_ijm1,P_ijp1,T_ij,T_im1j,T_ip1j,T_ijm1,T_ijp1,
      V_x_ij,V_x_im1j,V_x_ip1j,V_x_ijm1,V_x_ijp1,
      V_y_ij,V_y_im1j,V_y_ip1j,V_y_ijm1,V_y_ijp1 real=true
@syms Jx_im1j,Jx_ip1j,Jy_ijm1,Jy_ijp1,dx,dy real=true

Jx_im1j = -((P_im1j+P_ij)*V_x_ij+(T_im1j+T_ij)*(P_ij-Pim1j)/dx)/2
Jx_ip1j = -((P_im1j+P_ij)*V_x_ij+(T_im1j+T_ij)*(P_ij-Pim1j)/dx)/2  # TODO fix.

density_flux = (Jx_im1j-Jx_ip1j)/dy+(Jyijm1-Jyijp1)/dx
