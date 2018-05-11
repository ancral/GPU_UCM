#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <sys/time.h>
#include <sys/resource.h>

double get_time(){
	static struct timeval 	tv0;
	double time_, time;

	gettimeofday(&tv0,(struct timezone*)0);
	time_=(double)((tv0.tv_usec + (tv0.tv_sec)*1000000));
	time = time_/1000000;
	return(time);
}


typedef struct { float m, x, y, z, vx, vy, vz; } body;

void randomizeBodies(body *data, int n) {
	int i;
	for (i = 0; i < n; i++) {
		data[i].m  = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;

		data[i].x  = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
		data[i].y  = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
		data[i].z  = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;

		data[i].vx = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
		data[i].vy = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
		data[i].vz = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
	}
}
//#pragma acc routine 
void bodyForce(body *p, float dt, int n) {
	float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;
	int i, j;
	#pragma acc kernels loop collapse(2) independent present(p[0:n])
	for (i = 0; i < n; i++) { 
		for (j = 0; j < n; j++) {
			if (i!=j) {
				float dx = p[j].x - p[i].x;
				float dy = p[j].y - p[i].y;
				float dz = p[j].z - p[i].z;
				float distSqr = dx*dx + dy*dy + dz*dz;
				float invDist = 1.0f / sqrtf(distSqr);
				float invDist3 = invDist * invDist * invDist;

				float G = 6.674e-11;
				float g_masses = G * p[j].m * p[i].m;
				
				Fx += g_masses * dx * invDist3; 
				Fy += g_masses * dy * invDist3; 
				Fz += g_masses * dz * invDist3;
			}
		}

		p[i].vx += dt*Fx/p[i].m; p[i].vy += dt*Fy/p[i].m; p[i].vz += dt*Fz/p[i].m;

		Fx = 0.0f;  Fy = 0.0f;  Fz = 0.0f;
	}
}
//#pragma acc routine 
void integrate(body *p, float dt, int n){
	int i;
	#pragma acc kernels loop independent present(p[0:n])
	for (i = 0 ; i < n; i++) {
		p[i].x += p[i].vx*dt;
		p[i].y += p[i].vy*dt;
		p[i].z += p[i].vz*dt;
	}
}

int main(const int argc, const char** argv) {

	int nBodies = 1000;
	int iter;
	double totalTime, t0;
	if (argc > 1) nBodies = atoi(argv[1]);

	const float dt = 0.01f; // time step
	const int nIters = 100;  // simulation iterations

	body *p = (body*)malloc(nBodies*sizeof(body));

	randomizeBodies(p, nBodies); // Init pos / vel data
	#pragma acc data pcopy(p[0:nBodies])
	{
		t0 = get_time();
		for (iter = 1; iter <= nIters; iter++) {
			bodyForce(p, dt, nBodies); // compute interbody forces
			integrate(p, dt, nBodies); // integrate position
		}

		totalTime = get_time()-t0; 
	}
	printf("%d Bodies with %d iterations: %0.3f Millions Interactions/second\n", nBodies, nIters, 1e-6 * nBodies * nBodies / totalTime);

	free(p);
}
