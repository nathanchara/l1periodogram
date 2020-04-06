! --------------------------------------------------------------------------
! gglasso.f90: the BMD algorithm for group-lasso penalized learning.
! --------------------------------------------------------------------------
! 
! USAGE:
! 
! SUBROUTINE ls_f (bn,bs,ix,iy,gam,nobs,nvars,x,y,pf,dfmax,pmax,nlam,flmin,ulam,&
!                     eps,maxit,intr,nalam,b0,beta,idx,nbeta,alam,npass,jerr)
! 
! INPUT ARGUMENTS:
!    bn = number of groups
!    bs(bn) = size of each group
!    ix(bn) = first index for each group
!    iy(bn) = last index for each group
!    gam(bn) = upper bound gamma_k in MM algorithm
!    nobs = number of observations
!    nvars = number of predictor variables
!    x(nobs, nvars) = matrix of predictors, of dimension N * p; each row is an observation vector.
!    y(nobs) = response variable. This argument should be in {-inf, inf} for regression. 
!                and should be a two-level factor {-1, 1} for classification.
!    pf(bn) = relative penalties for each group
!                pf(k) = 0 => kth group unpenalized
!    dfmax = limit the maximum number of variables in the model.
!            (one of the stopping criterion)
!    pmax = limit the maximum number of variables ever to be nonzero. 
!           For example once beta enters the model, no matter how many 
!           times it exits or re-enters model through the path, it will 
!           be counted only once. 
!    nlam = the number of lambda values
!    flmin = user control of lambda values (>=0)
!            flmin < 1.0 => minimum lambda = flmin*(largest lambda value)
!            flmin >= 1.0 => use supplied lambda values (see below)
!    ulam(nlam) = user supplied lambda values (ignored if flmin < 1.0)
!    eps = convergence threshold for coordinate majorization descent. 
!          Each inner coordinate majorization descent loop continues 
!          until the relative change in any coefficient is less than eps.
!    maxit = maximum number of outer-loop iterations allowed at fixed lambda value. 
!            (suggested values, maxit = 100000)
!    intr = whether to include the intercept in the model
! 
!
! OUTPUT:
! 
!    nalam = actual number of lambda values (solutions)
!    b0(nvars) = intercept values for each solution
!    beta(nvars, nlam) = compressed coefficient values for each solution
!    idx(pmax) = pointers to compressed coefficients
!    nbeta(nlam) = number of compressed coefficients for each solution
!    alam(nlam) = lambda values corresponding to each solution
!    npass = actual number of passes over the data for all lambda values
!    jerr = error flag:
!           jerr  = 0 => no error
!           jerr > 0 => fatal error - no output returned
!                    jerr < 7777 => memory allocation error
!                    jerr = 10000 => maxval(vp) <= 0.0
!           jerr < 0 => non fatal error - partial output:
!                    Solutions for larger lambdas (1:(k-1)) returned.
!                    jerr = -k => convergence for kth lambda value not reached
!                           after maxit (see above) iterations.
!                    jerr = -10000-k => number of non zero coefficients along path
!                           exceeds pmax (see above) at kth lambda value.
! 
! LICENSE: GNU GPL (version 2 or later)
! 
! AUTHORS:
!    * Yi Yang (yi.yang6@mcgill.ca) and + Hui Zou (hzou@stat.umn.edu), 
!    * Department of Mathematics and Statistics, McGill University
!    + School of Statistics, University of Minnesota.
! 
! REFERENCES:
!    Yang, Y. and Zou, H. (2015). 
!    A Fast Unified Algorithm for Computing Group-Lasso Penalized Learning Problems
!    Statistics and Computing.
!    25(6), 1129-1141.

! --------------------------------------------------
SUBROUTINE ls_f (bn,bs,ix,iy,gam,nobs,nvars,x,y,beta0,pf,dfmax,pmax,nlam,flmin,ulam,&
                    eps,maxit,intr,nalam,b0,beta,idx,nbeta,alam,npass,jerr,verbose,scaley)
! --------------------------------------------------
    IMPLICIT NONE
    ! - - - arg types - - -
    DOUBLE PRECISION, PARAMETER :: big=9.9E30
    DOUBLE PRECISION, PARAMETER :: mfl = 1.0E-6
    INTEGER, PARAMETER :: mnlam = 6
    INTEGER:: mnl
    INTEGER:: bn
    INTEGER,intent(in)::bs(bn)
    INTEGER,intent(in)::ix(bn)
    INTEGER,intent(in)::iy(bn)
    INTEGER:: nobs
    INTEGER::nvars
    INTEGER,intent(in)::verbose
    INTEGER::dfmax
    INTEGER::pmax
    INTEGER::nlam
    INTEGER,intent(inout)::nalam
    INTEGER,intent(inout)::npass
    INTEGER,intent(inout)::jerr
    INTEGER::maxit
    INTEGER::intr
    INTEGER:: idx(pmax)
    INTEGER::nbeta(nlam)
    DOUBLE PRECISION:: flmin
    DOUBLE PRECISION,intent(in)::eps
    DOUBLE PRECISION::scaley
    DOUBLE PRECISION,intent(in):: x(nobs,nvars)
    DOUBLE PRECISION,intent(in)::y(nobs)
    DOUBLE PRECISION:: beta0(nvars)
    DOUBLE PRECISION::pf(bn)
    DOUBLE PRECISION::ulam(nlam)
    DOUBLE PRECISION::gam(bn)
    DOUBLE PRECISION:: b0(nlam)
    DOUBLE PRECISION::beta(nvars,nlam)
    DOUBLE PRECISION::alam(nlam)
    ! - - - local declarations - - -
    DOUBLE PRECISION:: max_gam
    DOUBLE PRECISION::d
    DOUBLE PRECISION::t
    DOUBLE PRECISION::dif
    DOUBLE PRECISION::unorm
    DOUBLE PRECISION::al
    DOUBLE PRECISION::alf
    DOUBLE PRECISION, DIMENSION (:), ALLOCATABLE :: b
    DOUBLE PRECISION, DIMENSION (:), ALLOCATABLE :: oldbeta
    DOUBLE PRECISION, DIMENSION (:), ALLOCATABLE :: r
    DOUBLE PRECISION, DIMENSION (:), ALLOCATABLE :: oldb
    DOUBLE PRECISION, DIMENSION (:), ALLOCATABLE :: u
    INTEGER, DIMENSION (:), ALLOCATABLE :: v
    DOUBLE PRECISION, DIMENSION (:), ALLOCATABLE :: dd
    INTEGER, DIMENSION (:), ALLOCATABLE :: oidx
    INTEGER:: g
    INTEGER::j
    INTEGER::l
    INTEGER::ni
    INTEGER::me
    INTEGER::start
    INTEGER::endindex
    INTEGER::it_count
    INTEGER::iteration_count
    ! - - - begin - - -
    ! - - - local declarations - - -
    DOUBLE PRECISION:: tlam
    INTEGER:: jx
    INTEGER:: jxx(bn)
    DOUBLE PRECISION:: ga(bn)
    DOUBLE PRECISION:: vl(nvars)
    DOUBLE PRECISION:: al0
! - - - allocate variables - - -
    ALLOCATE(b(0:nvars))
    ALLOCATE(oldbeta(0:nvars))
    ALLOCATE(r(1:nobs))
    ALLOCATE(oidx(1:bn))
!    PRINT *, 'in gglasso.f90: check 1'
! - - - checking pf - - -
    IF(maxval(pf) <= 0.0D0) THEN
        jerr=10000
        RETURN
    ENDIF
    pf=max(0.0D0,pf)
!    PRINT *, 'in gglasso.f90: check 2'
! - - - some initial setup - - -
    jxx = 0
    DO g = 1, bn
        ALLOCATE(u(bs(g)))
        u = beta0(ix(g):iy(g))
        IF(sqrt(dot_product(u,u))>0) THEN
            jxx(g) = 1
        ENDIF
        DEALLOCATE(u)
    ENDDO
!     PRINT *, 'in gglasso.f90: check 3'
    al = 0.0D0
    mnl = Min (mnlam, nlam)
    b(0)= 0.0D0
    b(1:nvars) = beta0
    oldbeta(0)= 0.0D0
    oldbeta(1:nvars) = beta0
    r = y-matmul(x,beta0)
    idx = 0
    oidx = 0
    npass = 0
    ni = npass
    alf = 0.0D0
    it_count = 0
!     PRINT *, 'in gglasso.f90: check 4'
! --------- lambda loop ----------------------------
    IF(flmin < 1.0D0) THEN
        flmin = Max (mfl, flmin)
        alf=flmin**(1.0D0/(nlam-1.0D0))
    ENDIF
    vl = matmul(r, x)/nobs
!    PRINT *, SHAPE(x)
!    DO g = 1,nvars
!        vl(g) = dot_product(r,x(1:nobs,g))/nobs
!    END DO 
!     PRINT *, 'in gglasso.f90: check 4a'
!    PRINT *, vl(1:2)	
    DO g = 1,bn
        ALLOCATE(u(bs(g)))
!        PRINT *,'allocation made'
!        PRINT *, ix(g), iy(g)	
!        PRINT *, vl(1:2)
        u = vl(ix(g):iy(g))
!        PRINT *,'u instantiated'
        ga(g) = sqrt(dot_product(u,u))
!        PRINT *,'dot product made'
        DEALLOCATE(u)
!        PRINT *,'u deallocated'
    END DO
!    PRINT *, 'in gglasso.f90: check 4b'
    DO l=1,nlam
        iteration_count=0
        IF(verbose==1) THEN
            print '("Lambda = ", f12.4)',ulam(l)/scaley
        END IF
        al0 = al
        IF(flmin>=1.0D0) THEN
            al=ulam(l)
        ELSE
            IF(l > 2) THEN
                al=al*alf
            ELSE IF(l==1) THEN
                al=big
            ELSE IF(l==2) THEN
                al0 = 0.0D0
                DO g = 1,bn
                    IF(pf(g)>0.0D0) THEN
                        al0 = max(al0, ga(g) / pf(g))
                    ENDIF
                END DO
                al = al0 * alf
            ENDIF
        ENDIF
        tlam = (2.0*al-al0)
        DO g = 1, bn
            IF(jxx(g) == 1) CYCLE
            IF(ga(g) > pf(g) * tlam) jxx(g) = 1
        ENDDO
!    PRINT *, 'in gglasso.f90: check 5'
! --------- outer loop ----------------------------
        DO
            oldbeta(0)=b(0)
            IF(ni>0) THEN
                DO j=1,ni
                    g=idx(j)
                    oldbeta(ix(g):iy(g))=b(ix(g):iy(g))
                ENDDO
            ENDIF
!    PRINT *, 'in gglasso.f90: check 6'
! --middle loop-------------------------------------
            DO
                npass=npass+1
                iteration_count=iteration_count+1
                dif=0.0D0
                DO g=1,bn
                    IF(jxx(g) == 0) CYCLE
                    start=ix(g)
                    endindex=iy(g)
                    ALLOCATE(u(bs(g)))
                    ALLOCATE(dd(bs(g)))
                    ALLOCATE(oldb(bs(g)))
                    oldb=b(start:endindex)
                    u=matmul(r,x(:,start:endindex))/nobs
                    u=gam(g)*b(start:endindex)+u
                    unorm=sqrt(dot_product(u,u))
                    t=unorm-pf(g)*al
                    IF(t>0.0D0) THEN
                        b(start:endindex)=u*t/(gam(g)*unorm)
                    ELSE
                        b(start:endindex)=0.0D0
                    ENDIF
                    dd=b(start:endindex)-oldb

                    IF(any(dd/=0.0D0)) THEN
                        dif=max(dif,gam(g)**2*dot_product(dd,dd))
                        r=r-matmul(x(:,start:endindex),dd)
                        IF(oidx(g)==0) THEN
                            ni=ni+1
                            IF(ni>pmax) EXIT
                            oidx(g)=ni
                            idx(ni)=g
                        ENDIF
                    ENDIF
                    DEALLOCATE(u,dd,oldb)
                ENDDO
                IF(intr /= 0) THEN
                    d=sum(r)/nobs
                    IF(d/=0.0D0) THEN
                        b(0)=b(0)+d
                        r=r-d
                        dif=max(dif,d**2)
                    ENDIF
                ENDIF
                IF (ni > pmax) EXIT
                IF (dif < eps**2) EXIT
                IF(npass > maxit) THEN
                    jerr=-l
                    RETURN
               ENDIF
!    PRINT *, 'in gglasso.f90: check 7'
! --inner loop----------------------
                DO
                    npass=npass+1
                    dif=0.0D0
                    DO j=1,ni
                        g=idx(j)
                        start=ix(g)
                        endindex=iy(g)
                        ALLOCATE(u(bs(g)))
                        ALLOCATE(dd(bs(g)))
                        ALLOCATE(oldb(bs(g)))
                        oldb=b(start:endindex)
                        u=matmul(r,x(:,start:endindex))/nobs
                        u=gam(g)*b(start:endindex)+u
                        unorm=sqrt(dot_product(u,u))
                        t=unorm-pf(g)*al
                        IF(t>0.0D0) THEN
                            b(start:endindex)=u*t/(gam(g)*unorm)
                        ELSE
                            b(start:endindex)=0.0D0
                        ENDIF
                        dd=b(start:endindex)-oldb
                        IF(any(dd/=0.0D0)) THEN
                            dif=max(dif,gam(g)**2*dot_product(dd,dd))
                            r=r-matmul(x(:,start:endindex),dd)
                        ENDIF
                        DEALLOCATE(u,dd,oldb)
                    ENDDO
                    IF(intr /= 0) THEN
                        d=sum(r)/nobs
                        IF(d/=0.0D0) THEN
                            b(0)=b(0)+d
                            r=r-d
                            dif=max(dif,d**2)
                        ENDIF
                    ENDIF
                    IF(dif<eps**2) EXIT
                    IF(npass > maxit) THEN
                        jerr=-l
                        RETURN
                    ENDIF
                ENDDO
            ENDDO
            IF(ni>pmax) EXIT
!    PRINT *, 'in gglasso.f90: check 8'
!--- final check ------------------------
            jx = 0
            max_gam = maxval(gam)
            IF(any((max_gam*(b-oldbeta)/(1+abs(b)))**2 >= eps**2)) jx = 1
            IF (jx /= 0) CYCLE
            vl = matmul(r, x)/nobs
            DO g = 1, bn
                IF(jxx(g) == 1) CYCLE
                ALLOCATE(u(bs(g)))
                u = vl(ix(g):iy(g))
                ga(g) = sqrt(dot_product(u,u))
                IF(ga(g) > al*pf(g))THEN
                    jxx(g) = 1
                    jx = 1
                    IF(verbose==1) THEN
                        print *,"    Strong rule failed, Group to add:",g
                    ENDIF
                ENDIF
                DEALLOCATE(u)
            ENDDO
            IF(jx == 1) CYCLE
            EXIT
        ENDDO
        IF(verbose==1) THEN
            ALLOCATE(u(2))
            ALLOCATE(v(bn))
            DO j=1,bn
                u=b(2*j:2*j+1)
                IF(dot_product(u,u)>0)THEN
                    it_count=it_count+1
                    v(it_count)=j
                END IF
            END DO
            DEALLOCATE(u)
            print *,"  --  Current active set: "
            write(*,10) v(1:it_count)
            10 format(10i10)
            print*,"  --  Number of iterations: ",iteration_count
            DEALLOCATE(v)
        ENDIF
        it_count=0
!    PRINT *, 'in gglasso.f90: check 9'
!---------- final update variable and save results------------
        IF(ni>pmax) THEN
            jerr=-10000-l
            EXIT
        ENDIF
        IF(ni>0) THEN
            DO j=1,ni
                g=idx(j)
                beta(ix(g):iy(g),l)=b(ix(g):iy(g))
            ENDDO
        ENDIF
        nbeta(l)=ni
        b0(l)=b(0)
        alam(l)=al
        nalam=l
        IF (l < mnl) CYCLE
        me=0
        DO j=1,ni
            g=idx(j)
            IF(any(beta(ix(g):iy(g),l)/=0.0D0)) me=me+1
        ENDDO
        IF(me>dfmax) EXIT
    ENDDO
    DEALLOCATE(b,oldbeta,r,oidx)
    RETURN
!    PRINT *, 'in gglasso.f90: check 10'
END SUBROUTINE ls_f




