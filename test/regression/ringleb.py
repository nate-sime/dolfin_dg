import numpy as np

__author__ = 'njcs4'


def fun_ringleb(gamma, x, y, c):
    gb = gamma - 1.0
    rho = c ** (2.0 / gb)
    jval = 1.0 / c + 1.0 / (3.0 * c ** 3) + 1.0 / (5.0 * c ** 5) - 0.5 * np.log((1.0 + c) / (1.0 - c))
    q2 = 2.0 * (1.0 - c ** 2) / gb

    return (x - jval / 2.0) ** 2 + y ** 2 - 1.0 / (4.0 * rho ** 2 * q2 ** 2)


def cspeed_ringleb(gamma, x, y):
    tol = 1.e-15
    clow = .5000000
    chigh = .9999999

    nit = 0
    err = 1.0
    while (nit <= 100 and err > tol):
        cnew = (clow + chigh) / 2.0
        if (fun_ringleb(gamma, x, y, cnew) * fun_ringleb(gamma, x, y, chigh) > 0.0):
            chigh = cnew
        else:
            clow = cnew

        nit = nit + 1
        err = abs(chigh - clow)

    if (err > tol):
        raise Exception('Error: cspeed_ringleb\n'
                        'Error tolerance was not achieved in bisection iteration'
                        'err = %.3e' % err)

    return cnew


def ringleb_anal_soln(x, y):
    gamma = 1.4
    c = cspeed_ringleb(gamma, x, y)

    gb = gamma - 1.0

    jval = 1.0 / c + 1.0 / (3.0 * c ** 3) + 1.0 / (5.0 * c ** 5) - 0.5 * np.log((1.0 + c) / (1.0 - c))
    rho = c ** (2.0 / gb)
    q2 = 2.0 * (1.0 - c ** 2) / gb
    qval = np.sqrt(q2)
    kval = np.sqrt(2.0 / (1.0 / q2 - 2.0 * rho * (x - jval / 2.0)))

    v = q2 / kval
    u = np.sqrt(abs(q2 - v * v))
    p = c * c * rho / gamma
    en = p / gb + 0.5 * rho * (u * u + v * v)

    return rho, rho * u, rho * v, en


def ringleb_boundary_shape(wght, x0, y0, x1, y1):
    nrelax = 300
    tol = 1.e-11

    gamma = 1.4
    gb = gamma - 1.0

    c0 = cspeed_ringleb(gamma, x0, y0)
    jval0 = 1.0 / c0 + 1.0 / (3.0 * c0 ** 3) + 1.0 / (5.0 * c0 ** 5) - 0.5 * np.log((1.0 + c0) / (1.0 - c0))
    rho0 = c0 ** (2.0 / gb)
    q0 = np.sqrt(2.0 * (1.0 - c0 ** 2) / gb)
    k0 = np.sqrt(2.0 / (1.0 / (q0 * q0) - 2.0 * rho0 * (x0 - jval0 / 2.0)))

    c1 = cspeed_ringleb(gamma, x1, y1)
    jval1 = 1.0 / c1 + 1.0 / (3.0 * c1 ** 3) + 1.0 / (5.0 * c1 ** 5) - 0.5 * np.log((1.0 + c1) / (1.0 - c1))
    rho1 = c1 ** (2.0 / gb)
    q1 = np.sqrt(2.0 * (1.0 - c1 ** 2) / gb)
    k1 = np.sqrt(2.0 / (1.0 / (q1 * q1) - 2.0 * rho1 * (x1 - jval1 / 2.0)))

    ratio = wght
    error = 1.0
    nit = 0

    while (error > tol and nit <= nrelax):
        nit = nit + 1
        qi = (1.0 - ratio) * q0 + ratio * q1
        ki = (1.0 - ratio) * k0 + ratio * k1
        ci = np.sqrt(1.0 - 0.5 * gb * qi * qi)
        jvali = 1.0 / ci + 1.0 / (3.0 * ci ** 3) + 1.0 / (5.0 * ci ** 5) - 0.5 * np.log((1.0 + ci) / (1.0 - ci))
        rhoi = ci ** (2.0 / gb)
        xi = jvali / 2.0 + 1.0 / (2.0 * rhoi) * (1.0 / (qi * qi) - 2.0 / (ki * ki))
        yi = 1.0 / (ki * qi * rhoi) * np.sqrt(1.0 - (qi / ki) ** 2)
        alen0 = np.sqrt((xi - x0) ** 2 + (yi - y0) ** 2)
        alen1 = np.sqrt((xi - x1) ** 2 + (yi - y1) ** 2)
        ratio2 = alen0 / (alen0 + alen1) + 1.e-12
        ratio = ratio * wght / ratio2
        error = abs(ratio2 - wght)
        ratio = min(.9999999999990, max(1.e-12, ratio))

    if (error > tol):
        raise Exception('Iteration failed in ringleb_boundary_shape\n '
                        'nit=%d\n '
                        'error=%.3e\n '
                        'tol=%.3e')

    return xi, yi

def mapv(iarg,jarg,ni,nj):

    if(iarg != 1 and iarg != ni and jarg != 1 and jarg != nj):
        mapv = (jarg-2)*(ni-2) + iarg-1 + 2*ni + 2*nj - 4
    else:
      if(jarg == 1):
        mapv = iarg
      elif( iarg == ni):
        mapv = ni + jarg - 1
      elif( jarg == nj):
        mapv = ni - iarg + ni + nj - 1
      else:
        mapv = nj - jarg + 2*ni + nj - 2

    return mapv


def gen_ringleb_vertices(gamma,gb,nk,nq,num_verts):

    max_nq_nk = max(nq,nk)
    
    xx, yy, ds = np.zeros(max_nq_nk), np.zeros(max_nq_nk), np.zeros(max_nq_nk)
    qr,ql,qq,dq,kr,kl = np.zeros(nq), np.zeros(nq), np.zeros(nq), np.zeros(nq), np.zeros(nq), np.zeros(nq)
    kt,kb,dk,qt,qb = np.zeros(nk), np.zeros(nq), np.zeros(nq), np.zeros(nq), np.zeros(nq)
    x,y = np.zeros(num_verts), np.zeros(num_verts)

    nrelax = 100

    # fixed parameters for ringleb geometry

    kmin = .60 
    kmax = .98
    qmin = .43
    theta_inflow = 88.0*np.pi/180.0

    kval = kmin
    qval = kmin*np.sin(theta_inflow)
    c = np.sqrt( 1.0 - 0.5*gb*qval*qval )
    rho = c**(2.0/gb)
    jval = 1.0/c + 1.0/(3.0*c**3) + 1.0/(5.0*c**5) - 0.5*np.log( (1.0+c)/(1.0-c) )
    xl = jval/2.0 + 1.0/(2.0*rho)*(1.0/(qval**2)-2.0/(kval**2))
    yl = np.cos(theta_inflow)/(rho*kval*qval)
    qlb = qval

    kval = kmax
    qval = kmax*np.sin(theta_inflow)
    c = np.sqrt( 1.0 - 0.5*gb*qval*qval )
    rho = c**(2.0/gb)
    jval = 1.0/c + 1.0/(3.0*c**3) + 1.0/(5.0*c**5) - 0.5*np.log( (1.0+c)/(1.0-c) )
    xr = jval/2.0 + 1.0/(2.0*rho)*(1.0/(qval**2)-2.0/(kval**2))
    yr = np.cos(theta_inflow)/(rho*kval*qval)
    qrb = qval

    # evaluate solution along a-b

    for n in range(1,nk+1):
      ss = float(n-1)/float(nk-1)
      coords = ringleb_boundary_shape(ss,xl,yl,xr,yr)
      x[mapv(n,1,nk,nq)] = coords[0]
      y[mapv(n,1,nk,nq)] = coords[1]
      soln = ringleb_anal_soln(coords[0],coords[1])
      rho = soln[0]
      uvel = soln[1]/rho
      vvel = soln[2]/rho
      qb[n] = np.sqrt(uvel**2+vvel**2)
      kb[n] = (uvel**2+vvel**2)/vvel

# segment b-c  (psi = 1/kmax)

    kval = kmax
    for n in range(1,nq+1):
      qr[n] = qrb + (qmin-qrb)*float(n-1)/float(nq-1)

    for nit in range(1, nrelax+1):

      for n in range(1, nq+1):
        qval = qr[n]
        # qval = min(qval,kval)
        psi = 1.0/kmax
        theta = np.asin( psi*qval )
        u = qval*np.cos(theta)
        v = qval*np.sin(theta)
        c = np.sqrt( 1.0 - 0.5*gb*qval*qval )
        rho = c**(2.0/gb)
        jval = 1.0/c + 1.0/(3.0*c**3) + 1.0/(5.0*c**5) - 0.5*np.log( (1.0+c)/(1.0-c) )
        xx[n] = jval/2.0 + 1.0/(2.0*rho)*(1.0/(qval*qval) - 2.0/(kval*kval) )
        yy[n] = 1.0/(kval*qval*rho)*np.sqrt(1.0 - (qval/kval)**2 )

      dsave = 0.0
      for n in range(1, nq):
        dq[n] = qr[n+1]-qr[n]
        ds[n] = np.sqrt( (xx[n+1]-xx[n])**2 + (yy[n+1]-yy[n])**2 )
        dsave = dsave + ds[n]
         
      dsave = dsave/float(nq-1)
         
      for n in range(1,nq):
        dq[n] = dq[n]*(dsave/ds[n] + 1.0)/2.0
         
      sum = 0.0
      for n in range(1,nq):
        sum = sum + dq[n]
         
      rat = (qr[nq]-qr[0])/sum
         
      for n in range(1, nq):
        qr[n+1] = qr[n] + dq[n]*rat

# evaluate solution along b-c

    kr[0] = kmax
    for n in range(1,nq+1):
      qval = qr[n]
      # qval = min(qval,kval)
      psi = 1.0/kmax
      theta = np.asin( psi*qval )
      u = qval*np.cos(theta)
      v = qval*np.sin(theta)
      c = np.sqrt( 1.0 - 0.5*gb*qval*qval )
      rho = c**(2.0/gb)
      jval = 1.0/c + 1.0/(3.0*c**3) + 1.0/(5.0*c**5) - 0.5*np.log( (1.0+c)/(1.0-c) )
      coords[0] = jval/2.0 + 1.0/(2.0*rho)*(1.0/(qval*qval) - 2.0/(kval*kval) )
      coords[1] = 1.0/(kval*qval*rho)*np.sqrt(1.0 - (qval/kval)**2 )
      kr[n] = kmax
      x[mapv(nk,n,nk,nq)] = coords[0]
      y[mapv(nk,n,nk,nq)] = coords[1]

# evaluate solution along c-d

    for n in range(1,nk+1):
      kt[n] = kmin + (kmax-kmin)*float(n-1)/float(nk-1)
      qt[n] = qmin

    for nit in range(1, nrelax+1):
      for n in range(1, nk+1):
        kval = kt[n]
        qval = qmin
        # qval = min(qval,kval)
        psi = 1.0/kval
        theta = np.asin( psi*qval )
        u = qval*np.cos(theta)
        v = qval*np.sin(theta)
        c = np.sqrt( 1.0 - 0.5*gb*qval*qval )
        rho = c**(2.0/gb)
        jval = 1.0/c + 1.0/(3.0*c**3) + 1.0/(5.0*c**5) - 0.5*np.log( (1.0+c)/(1.0-c) )
        xx[n] = jval/2.0 + 1.0/(2.0*rho)*(1.0/(qval*qval) - 2.0/(kval*kval) )
        yy[n] = 1.0/(kval*qval*rho)*np.sqrt(1.0 - (qval/kval)**2)

      dsave = 0.0
      for n in range(1, nk):
        dk[n] = kt[n+1]-kt[n]
        ds[n] = np.sqrt( (xx[n+1]-xx[n])**2 + (yy[n+1]-yy[n])**2 )
        dsave = dsave + ds[n]
         
      dsave = dsave/float(nk-1)
         
      for n in range(1, nk):
        dk[n] = dk[n]*(dsave/ds[n] + 1.0)/2.0
         
      sum = 0.0
      for n in range(1, nk):
        sum = sum + dk[n]

      rat = (kt[nk]-kt[0])/sum
         
      for n in range(1,nk):
        kt[n+1] = kt[n] + dk[n]*rat

    # use bottom k

    for n in range(1, nk+1):
      kt[n] = kb[n]

    for n in range(2, nk+1):
      kval = kt[nk-n+1]
      qval = qmin
      # qval = min(qval,kval)
      psi = 1.0/kval
      theta = np.asin( psi*qval )
      u = qval*np.cos(theta)
      v = qval*np.sin(theta)
      c = np.sqrt( 1.0 - 0.5*gb*qval*qval )
      rho = c**(2.0/gb)
      jval = 1.0/c + 1.0/(3.0*c**3) + 1.0/(5.0*c**5) - 0.5*np.log( (1.0+c)/(1.0-c) )
      coords[0] = jval/2.0 + 1.0/(2.0*rho)*(1.0/(qval*qval) - 2.0/(kval*kval) )
      coords[1] = 1.0/(kval*qval*rho)*np.sqrt(1.0 - (qval/kval)**2)
      x[mapv(nk-n+1,nq,nk,nq)] = coords[0]
      y[mapv(nk-n+1,nq,nk,nq)] = coords[1]

    # segment d-a  (psi = 1/kmin)

    kval = kmin
    for n in range(1, nq+1):
      ql[n] = qlb + (qmin-qlb)*float(n-1)/float(nq-1)

    for nit in range(1, nrelax+1):
      
      for n in range(1, nq+1):
        qval = ql[n]
        # qval = min(qval,kval)
        psi = 1.0/kval
        theta = np.asin( psi*qval )
        u = qval*np.cos(theta)
        v = qval*np.sin(theta)
        c = np.sqrt( 1.0 - 0.5*gb*qval*qval )
        rho = c**(2.0/gb)
        jval = 1.0/c + 1.0/(3.0*c**3) + 1.0/(5.0*c**5) - 0.5*np.log( (1.0+c)/(1.0-c) )
        xx[n] = jval/2.0 + 1.0/(2.0*rho)*(1.0/(qval*qval) - 2.0/(kval*kval) )
        yy[n] = 1.0/(kval*qval*rho)*np.sqrt(1.0 - (qval/kval)**2 )

      dsave = 0.0
      for n in range(1, nq):
        dq[n] = ql[n+1]-ql[n]
        ds[n] = np.sqrt( (xx[n+1]-xx[n])**2 + (yy[n+1]-yy[n])**2 )
        dsave = dsave + ds[n]

      dsave = dsave/float(nq-1)

      for n in range(1, nq):
        dq[n] = dq[n]*(dsave/ds[n] + 1.0)/2.0

      sum = 0.0
      for n in range(1, nq):
        sum = sum + dq[n]

      rat = (ql[nq]-ql[0])/sum

      for n in range(1,nq):
        ql[n+1] = ql[n] + dq[n]*rat




# evaluate solution along d-a

    kl[0] = kval
    kl[nq] = kval
    for n in range(2, nq):
      qval = ql[nq-n+1]
      kl[nq-n+1] = kval
      # qval = min(qval,kval)
      psi = 1.0/kval
      theta = np.asin( psi*qval )
      u = qval*np.cos(theta)
      v = qval*np.sin(theta)
      c = np.sqrt( 1.0 - 0.5*gb*qval*qval )
      rho = c**(2.0/gb)
      jval = 1.0/c + 1.0/(3.0*c**3) + 1.0/(5.0*c**5) - 0.5*np.log( (1.0+c)/(1.0-c) )
      coords[0] = jval/2.0 + 1.0/(2.0*rho)*(1.0/(qval*qval) - 2.0/(kval*kval) )
      coords[1] = 1.0/(kval*qval*rho)*np.sqrt(1.0 - (qval/kval)**2 )
      x[mapv(1,nq-n+1,nk,nq)] = coords[0]
      y[mapv(1,nq-n+1,nk,nq)] = coords[1]



    # big loop over the rest of the field
      
    for m in range(2, nk):

      kval = kt[m]

      for n in range(1, nq+1):
        qq[n] = qt[m] + (qb[m]-qt[m])*float(n-1)/float(nq-1)


      for nit in range(1,nrelax+1):
      
        for n in range(1,nq+1):
          qval = qq[n]
          qval = min(qval,kval)
          psi = 1.0/kval
          theta = np.asin( psi*qval )
          u = qval*np.cos(theta)
          v = qval*np.sin(theta)
          c = np.sqrt( 1.0 - 0.5*gb*qval*qval )
          rho = c**(2.0/gb)
          jval = 1.0/c + 1.0/(3.0*c**3) + 1.0/(5.0*c**5) - 0.5*np.log( (1.0+c)/(1.0-c) )
          xx[n] = jval/2.0 + 1.0/(2.0*rho)*(1.0/(qval*qval) - 2.0/(kval*kval) )
          yy[n] = 1.0/(kval*qval*rho)*np.sqrt(1.0 - (qval/kval)**2)

        dsave = 0.0
        for n in range(1, nq):
          dq[n] = qq[n+1]-qq[n]
          ds[n] = np.sqrt( (xx[n+1]-xx[n])**2 + (yy[n+1]-yy[n])**2 )
          dsave = dsave + ds[n]

        dsave = dsave/float(nq-1)
            
        for n in range(1, nq):
          dq[n] = dq[n]*( dsave/ds[n] + 1.0)/2.0

        sum = 0.0
        for n in range(1, nq):
          sum = sum + dq[n]

        rat = (qq[nq]-qq[0])/sum

        for n in range(1, nq):
          qq[n+1] = qq[n] + dq[n]*rat

    # output field solution here
     
      for n in range(2, nq):
        qval = qq[n]
        qval = min(qval,kval)
        psi = 1.0/kval
        theta = np.asin( psi*qval )
        u = qval*np.cos(theta)
        v = qval*np.sin(theta)
        c = np.sqrt( 1.0 - 0.5*gb*qval*qval )
        rho = c**(2.0/gb)
        jval = 1.0/c + 1.0/(3.0*c**3) + 1.0/(5.0*c**5) - 0.5*np.log( (1.0+c)/(1.0-c) )
        x[mapv(m,nq-n+1,nk,nq)] = jval/2.0 + 1.0/(2.0*rho)*(1.0/(qval*qval) - 2.0/(kval*kval) )
        y[mapv(m,nq-n+1,nk,nq)] = 1.0/(kval*qval*rho)*np.sqrt(1.0 -(qval/kval)**2)

    # Finally, store the mesh vertices in mesh_data

  #   for i in range(1, num_verts+1):
  #     coords[0] = x[i]
  #     coords[1] = y[i]
  #     call addto_coordinates(coords,i,2,mesh_data)
  #
  #
  #   deallocate(xx,yy,ds)
  #
  # end subroutine gen_ringleb_vertices

print gen_ringleb_vertices(1.4, 1.4-1.0, 2, 2, 4)