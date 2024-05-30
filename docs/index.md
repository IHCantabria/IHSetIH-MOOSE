# IHSetIH-MOOSE

## Summary

Jaramillo et al. (2021b) proposed a hybrid model for embayed beaches, IH-MOOSE (Model Of Shoreline Evolution), by integrating cross-shore, planform and rotation equilibrium-based evolution models. In other words, IH-MOOSE combines the following models: (1) equilibrium cross-shore evolution model (Yates et al., 2009), (2) the equilibrium planform model (Hsu and Evans, 1989) and (3) the equilibrium shoreline rotation model (Jaramillo et al., 2021a).

## Model formula

Jaramillo et al. (2021) proposed a hybrid shoreline evolution model based on cross-shore, planform and rotation equilibrium models. The hybrid models suggested by Jaramillo et al. (2021) can be expressed as follow:

```text
(1) Equilibrium cross-shore evolution model (Yates et al., 2009):
(∂S(t))/∂t=C^±∙E^(1/2)(E-E_eq (S))

	E : the incoming breaking wave energy related to the breaking wave height (water depth at breaking h_b and breaking index γ) as E=(H_b/4.004)^2=(γ/4.004)^2 h_b^2
	E_eq : the equilibrium wave energy corresponding to the current shoreline position
	S(t) : the shoreline position at time t
	S_eq : the equilibrium shoreline position (S_eq=(E-b)/a; a,b : the parameters satisfy the energy equilibrium function)
	C^± : the free parameters where C^+ indicates the accretion rate and C^- indicates the erosion rate, respectively

(2) Equilibrium planform model (Hsu and Evans, 1989):
R/R_o =C_0+C_1 (β/θ)+C_2 (β/θ)^2

	R : the radius measured from the tip of the headland breakwater
	R_o : the length of the control line joining the updrift diffraction point to the down-coast control point
	θ : location on the shoreline at an angle measured from the wave crest
	β : the angle between the control line and the wavefront at the diffraction point
	C_i : the calibration parameters that depend on the wave obliquity (β) based on measured shapes of the model beaches (i=0,1 and 2)

(3) Equilibrium shoreline rotation model (Jaramillo et al., 2021)
Jaramillo et al. (2021) suggested a shoreline rotation model that predicts the temporal evolution of the shoreline orientation based on the concept of previous research and the observation data as follows:
(∂α_s (t))/∂t=L^±∙P(α_s-α_eq )

	P : the incoming wave power related to the significant wave height H_s and the wave peak period T_p as P=H_s^2∙T_p
	α_eq : the asymptotical equilibrium shoreline orientation (α_eq=(θ-b')/a';θ : the incident wave direction, a',b' : the empirical parameters satisfy the linear relationship)
	α_s (t) : the shoreline orientation at time t
	L^± : the proportional constants where L^+ indicates the clockwise shoreline rotation and L^- indicates the counterclockwise rotation, respectively
```