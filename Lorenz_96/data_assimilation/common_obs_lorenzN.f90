MODULE common_obs
!=======================================================================
!
! [PURPOSE:] Observation operator for data assimilation
!
!=======================================================================
!$USE OMP_LIB
  USE common_tools

  IMPLICIT NONE

  PUBLIC

CONTAINS
!=======================================================================
! Model_to_Obs
! This routine assumes equispaced state variables in x and t.
! Observations can be arbitrarily distributed in x and t.
! Cyclic boundary conditions are assumed in space.
!=======================================================================
SUBROUTINE model_to_obs(nx,no,nt,nens,obsloc,x,xloc,tloc,obs,obstype,valid_obs)

IMPLICIT NONE
INTEGER,INTENT(IN)         :: no , nx , nt , nens             !Number of observations , number of state variables , number of times , number of ensemble members
REAL(r_size),INTENT(IN)    :: obsloc(no,2)                    !Observation location  (space,time)
INTEGER,INTENT(IN)         :: obstype(no)                     !Observation type. 1 - Observe X , 2 - Observe X**2
REAL(r_size),INTENT(IN)    :: x(nx,nens,nt)                   !State variables (space, ensemble, time)
REAL(r_size),INTENT(IN)    :: xloc(nx) , tloc(nt)             !Space grid point locations, time grid point locations for the model output.
REAL(r_size),INTENT(OUT)   :: obs(no,nens)                    !State in observation space.

INTEGER                    :: io , iloc, ie , it , ix         !Loop indices
REAL(r_size)               :: rx, rt, dx,dt                   !Auxiliary variables
INTEGER                    :: ixloc , itloc                   !Auxiliary variables
REAL(r_size)               :: tmp_x(nx+2,nens,nt),tmp_xloc(nx+2) !Temporal array to use cyclic boundary conditions.
REAL(r_size)               :: tmp_obs(2,2,nens)               !Temporal array for space-time interpolation.

INTEGER,INTENT(OUT)        :: valid_obs(no)                   !Mask to indicate if the observation was within or outside the model domain (space/time)
                                                              !1 means a valid observation , 0 that the observation is not valid.


!Assuming regular grid in space and time.
dx=xloc(2)-xloc(1)
dt=1.0d0

IF( nt > 1 )THEN
  dt=tloc(2)-tloc(1)
ENDIF

!Define x_tmp in order to assume cyclic boundary conditions.
tmp_x(2:nx+1,:,:)=x(:,:,:)
tmp_x(1,:,:)     =x(nx,:,:)
tmp_x(nx+2,:,:)  =x(1 ,:,:)

tmp_xloc(2:nx+1)=xloc
tmp_xloc(1)=xloc(1)-dx
tmp_xloc(nx+2)=xloc(nx)+dx

!Initialize tmp_obs
tmp_obs=0.0d0

!Initialize valid_obs
valid_obs=1


!Initialize obs
obs=0.0d0

DO io=1,no
  !Compute the position of the observation with respect to the grid points in space and time.
  rx=( ( obsloc(io,1) - tmp_xloc(1) )/dx ) + 1.0d0
  ixloc= FLOOR( rx ) 
  IF( nt > 1 )THEN
    rt=( ( obsloc(io,2) - tloc(1) )/dt ) + 1.0d0
    itloc= FLOOR( rt ) 
  ELSE
    !If we have only one time we round the observation time to the closest time.
    itloc= 1.0d0
    rt=1.0d0
  ENDIF

 !Observation operator for observations in between time and space grid points.
 !2D linear interpolation is applied in this case.

 !IF( ixloc /= obsloc(io,1) .or. itloc /= obsloc(io,2) )THEN
  IF( ixloc >= 1 .and. ixloc <= nx+1 .or. itloc >= 1 .and. itloc <= nt )THEN
   
   !The location of the observation is in between grid points, perform linear interpolation.
   IF( rx == ixloc .and. rt == itloc )THEN
     !Observation are at model output times and at the location of space grid points.
     tmp_obs(1,1,:)=tmp_x(ixloc,:, itloc)
     tmp_obs(1,2,:)=tmp_x(ixloc,:, itloc)
     tmp_obs(2,1,:)=tmp_x(ixloc,:, itloc)
     tmp_obs(2,2,:)=tmp_x(ixloc,:, itloc)
   ELSEIF( rx == ixloc .and. rt /= itloc )THEN
     !Observations are at the location of space grid points but at arbitrary times.
     !Time and space interpolation.
     tmp_obs(1,1,:)=tmp_x(ixloc,:, itloc  )
     tmp_obs(1,2,:)=tmp_x(ixloc,:, itloc+1)
     tmp_obs(2,1,:)=tmp_x(ixloc,:, itloc  )
     tmp_obs(2,2,:)=tmp_x(ixloc,:, itloc+1)
   ELSEIF( rt == itloc .and. rx /= ixloc )THEN
     !Observations are at the model output time.
     !Only space interpolation is required.
     tmp_obs(1,1,:)=tmp_x(ixloc  ,:, itloc)
     tmp_obs(1,2,:)=tmp_x(ixloc  ,:, itloc)
     tmp_obs(2,1,:)=tmp_x(ixloc+1,:, itloc)
     tmp_obs(2,2,:)=tmp_x(ixloc+1,:, itloc)
   ELSE
     !Observations are at arbitrary times.
     !Time and space interpolation.
     tmp_obs(1,1,:)=tmp_x(ixloc  ,:, itloc  )
     tmp_obs(1,2,:)=tmp_x(ixloc  ,:, itloc+1)
     tmp_obs(2,1,:)=tmp_x(ixloc+1,:, itloc  )
     tmp_obs(2,2,:)=tmp_x(ixloc+1,:, itloc+1)
   ENDIF


   rx= rx - ixloc + 1.0d0
   rt= rt - itloc + 1.0d0

   !WRITE(*,*)rx,rt

   DO ie=1,nens
 
    IF( obstype(io) == 1 )THEN
      CALL itpl_2d(tmp_obs(:,:,ie),rx,rt,obs(io,ie),2,2)
    ELSEIF( obstype(io) == 2 )THEN
      CALL itpl_2d(tmp_obs(:,:,ie)**2,rx,rt,obs(io,ie),2,2)
    ELSEIF( obstype(io) == 3 )THEN
      CALL itpl_2d(tmp_obs(:,:,ie),rx,rt,obs(io,ie),2,2)
      obs(io,ie) = obs(io,ie) !-1.0e0
      IF( obs(io,ie) > 0.001 )THEN
          !Pseudo reflectivity
          obs(io,ie)= 1.0e18 * 720 * ( (obs(io,ie)*1.0e-3) ** 1.75 )
          obs(io,ie) = obs(io,ie) / ( ( 3.14159 ** 1.75 ) * ( 1000.0e0 ** 0.75 ) * ( 8.0e6  ** 1.75 ) )
          obs(io,ie) = 10.0*log10( obs(io,ie) )  
      ELSE
          obs(io,ie)= 1.0e18 * 720 * ( (0.001*1.0e-3) ** 1.75 )
          obs(io,ie) = obs(io,ie) / ( ( 3.14159 ** 1.75 ) * ( 1000.0e0 ** 0.75 ) * ( 8.0e6  ** 1.75 ) )
          obs(io,ie) = 10.0*log10( obs(io,ie) )
    
      ENDIF
      IF ( obs(io,ie) < -30.0e0 )obs(io,ie) = -30.0e0
    ELSE

       WRITE(*,*)"ERROR: Not recognized observation type"

    END IF
 
   END DO

  ELSE  !The observation is outside the model domain (space/time)
 
    valid_obs(io)=0

  ENDIF

 !ENDIF

END DO

IF( SUM(valid_obs) < no )THEN
  WRITE(*,*)"[Warning]: The number of valid observations is lower than the number of input observations"
ENDIF

END SUBROUTINE model_to_obs 


!-----------------------------------------------------------------------
! Observation error
!-----------------------------------------------------------------------

SUBROUTINE add_obs_error(no,nens,obs,obsout,obs_error,obs_bias,otype)

IMPLICIT NONE
INTEGER,INTENT(IN)         :: no , nens                       !Number of observations , number of ensemble members
REAL(r_size),INTENT(IN)    :: obs(no,nens)                    !Observations without error.
REAL(r_size),INTENT(IN)    :: obs_error(no),obs_bias(no)      !Diagonal of R matrix and systematic observation error
INTEGER,INTENT(IN)         :: otype
REAL(r_size),INTENT(OUT)   :: obsout(no,nens)                 !State in observation space.

INTEGER                    :: io , ie
REAL(r_size)               :: randomn(no)
REAL(r_size)               :: tmp_error(no)

tmp_error=sqrt( obs_error )

obsout=obs

DO ie = 1,nens
  
  CALL com_randn( no , randomn , ie)
  DO io = 1,no
    IF( otype == 3 .and. obsout(io,ie) <= -30.0e0 )THEN
      obsout(io,ie) = obsout(io,ie)
    ELSE
      obsout(io,ie)=obsout(io,ie)+tmp_error(io)*randomn(io)+obs_bias(io) 
    ENDIF

    IF( otype == 3 .and. obsout(io,ie) <= -30.0e0 )THEN
      obsout(io,ie) = -30.0e0
    ENDIF

  END DO
END DO

END SUBROUTINE add_obs_error

!-----------------------------------------------------------------------
! Get observation number based on network type
!-----------------------------------------------------------------------
SUBROUTINE get_obs_number( ntype , nx , nt , time_density , space_density , no )
IMPLICIT NONE
CHARACTER(*), INTENT(IN)   :: ntype  !Network type (see options below)
INTEGER     , INTENT(IN)   :: nx,nt  !Number of grid points and times.
REAL(r_size), INTENT(IN)   :: time_density , space_density !Obs density. 
INTEGER     , INTENT(OUT)  :: no     !Number of observations
                              !1 means one observation per grid point or time. 0 means no observations.
INTEGER                    :: skipx , skipt , iobs , it , ix , tmpt , tmpx

IF( ntype == 'regular' .or. ntype == 'REGULAR' )THEN
 !Regular observation array.
 skipx=INT(1/space_density)
 skipt=INT(1/time_density)
 tmpx=0
 DO ix=1,nx,skipx
   tmpx=tmpx+1
 ENDDO
 tmpt=0
 DO it=1,nt,skipt
   tmpt=tmpt+1
 ENDDO
 no = tmpx * tmpt

ELSEIF( ntype == 'random' .or. ntype == 'RANDOM' )THEN
 !Random observation array.
 no=FLOOR( nx*space_density )*FLOOR( nt*time_density )

ELSEIF( ntype == 'fromfile' .or. ntype == 'FROMFILE' )THEN
!Input location from file.

  WRITE(*,*)"[Error]: FROMFILE option is not available yet"

ENDIF



END SUBROUTINE get_obs_number

!-----------------------------------------------------------------------
! Observation location
!-----------------------------------------------------------------------

SUBROUTINE get_obs_location( ntype , nx , nt , time_density , space_density , no , obsloc )
IMPLICIT NONE
CHARACTER(*), INTENT(IN)   :: ntype  !Network type (see options below)
INTEGER     , INTENT(IN)   :: nx,nt  !Number of grid points and times.
INTEGER     , INTENT(IN)   :: no     !Number of observations
REAL(r_size), INTENT(IN)   :: time_density , space_density !Obs density. 
                              !1 means one observation per grid point or time. 0 means no observations.
REAL(r_size), INTENT(OUT) :: obsloc(no,2)
INTEGER                    :: skipx , skipt , iobs , it , ix

IF( ntype == 'regular' .or. ntype == 'REGULAR' )THEN
 !Regular observation array.
 skipx=INT(1/space_density)
 skipt=INT(1/time_density)

 iobs=0
  DO it = 1,nt,skipt
   DO ix = 1,nx,skipx
     iobs=iobs+1
     obsloc(iobs,1)=REAL(ix,r_size)
     obsloc(iobs,2)=REAL(it,r_size)
   END DO
  END DO

ELSEIF( ntype == 'random' .or. ntype == 'RANDOM' )THEN
 !Random observation array.

 CALL com_rand( no , obsloc(:,1) )
 CALL com_rand( no , obsloc(:,2) )

 obsloc(:,1)=obsloc(:,1)*REAL(nx,r_size)                  !From 0 - nx
 obsloc(:,2)=obsloc(:,2)*(REAL(nx,r_size)-1.0d0) + 1.0d0  !From 1 - nt

ELSEIF( ntype == 'fromfile' .or. ntype == 'FROMFILE' )THEN
!Input location from file.

  WRITE(*,*)"[Error]: FROMFILE option is not available"

ENDIF

END SUBROUTINE get_obs_location

!-----------------------------------------------------------------------
! 2D bilinear interpolation
!-----------------------------------------------------------------------
SUBROUTINE itpl_2d(var,ri,rj,var5,nx,no)
  IMPLICIT NONE
  INTEGER     ,INTENT(IN) :: nx,no  !Grid dimensions
  REAL(r_size),INTENT(IN) :: var(nx,no)
  REAL(r_size),INTENT(IN) :: ri
  REAL(r_size),INTENT(IN) :: rj
  REAL(r_size),INTENT(OUT) :: var5
  REAL(r_size) :: ai,aj
  INTEGER :: i,j

  i = FLOOR(ri)
  ai = ri - REAL(i,r_size)
  j = FLOOR(rj)
  aj = rj - REAL(j,r_size)


  IF(i <= nx) THEN
    var5 = var(i,j) * (1-ai) * (1-aj) &
       & + var(i+1,j) *    ai  * (1-aj) &
       & + var(i,j+1  ) * (1-ai) *    aj  &
       & + var(i+1,j+1  ) *    ai  *    aj
  END IF

  RETURN
END SUBROUTINE itpl_2d



END MODULE common_obs
