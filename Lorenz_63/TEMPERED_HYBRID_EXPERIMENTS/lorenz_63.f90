module lorenz63

PUBLIC

CONTAINS

!
!! MAIN is the main program for LORENZ_ODE.
!  Licensing:
!    This code is distributed under the GNU LGPL license.
!  Modified:
!    14 October 2013
!  Author:
!    John Burkardt
!  Parameters:
!    None

subroutine forward_model( ne , x0 , p , nt , dt , xf )
  implicit none
  integer , parameter :: m = 3
  integer , intent(in) :: ne !number of ensemble members
  integer , intent(in) :: nt !number of time steps 
  real ( kind = 8 ) , intent(in) :: dt !time step
  real ( kind = 8 ) , intent(in) :: x0(m,ne) !Initial conditions
  real ( kind = 8 ) , intent(out) :: xf(m,ne) !Final condition
  real ( kind = 8 ) , intent(in) :: p(m)
  
  integer :: it , ie 
  real ( kind = 8 ) xi(m),xe(m)  !current model state

!
!  Compute the approximate solution at equally spaced times.
!
!$OMP PARALLEL DO SCHEDULE(DYNAMIC) PRIVATE(ie,xe,xi,it)
  do ie = 1 , ne  !loop over ensemble members
     xi = x0(:,ie) 
     do it = 1 , nt  !loop over time iterations
       call rk4vec (xi,dt,xe,p)
       xi=xe
     end do
     xf(:,ie)=xe !Store the final state after nt integration steps.
  end do
!$OMP END PARALLEL DO

  return

end subroutine forward_model

subroutine lorenz_rhs ( x, p , dxdt )

!*****************************************************************************80
!
!! LORENZ_RHS evaluates the right hand side of the Lorenz ODE.
!  Licensing:
!    This code is distributed under the GNU LGPL license.
!  Modified:
!    08 October 2013
!  Author:
!    John Burkardt
!
  implicit none
  integer , parameter :: m = 3
  real ( kind = 8 ) , intent(in) :: p(m)
  real ( kind = 8 ) , intent(out) :: dxdt(m)
  real ( kind = 8 ) , intent(in)  :: x(m)
  dxdt(1) = p(1) * ( x(2) - x(1) )
  dxdt(2) = x(1) * ( p(2) - x(3) ) - x(2)
  dxdt(3) = x(1) * x(2) - p(3) * x(3)
  return
end subroutine lorenz_rhs
subroutine rk4vec (  u0, dt, u , p )

!*****************************************************************************80
!
!! RK4VEC takes one Runge-Kutta step for a vector system.
!  Licensing:
!    This code is distributed under the GNU LGPL license. 
!  Modified:
!    08 October 2013
!  Author:
!    John Burkardt
  implicit none
  integer , parameter :: m = 3
  real ( kind = 8 ) , intent(in) :: dt
  real ( kind = 8 ) , intent(in) :: p(m) !model parameters
  real ( kind = 8 ) f0(m)
  real ( kind = 8 ) f1(m)
  real ( kind = 8 ) f2(m)
  real ( kind = 8 ) f3(m)
  real ( kind = 8 ) , intent(out) :: u(m)
  real ( kind = 8 ) , intent(in)  :: u0(m)
  real ( kind = 8 ) u1(m)
  real ( kind = 8 ) u2(m)
  real ( kind = 8 ) u3(m)
  call lorenz_rhs( u0, p, f0 )
  u1 = u0 + dt * f0 / 2.0D+00
  call lorenz_rhs(  u1 , p , f1 )
  u2 = u0 + dt * f1 / 2.0D+00
  call lorenz_rhs(  u2 , p , f2 )
  u3 = u0 + dt * f2
  call lorenz_rhs(  u1 , p , f3 )
  u = u0 + dt * ( f0 + 2.0D+00 * f1 + 2.0D+00 * f2 &
    + f3 ) / 6.0D+00

  return
end subroutine rk4vec

end module  lorenz63
