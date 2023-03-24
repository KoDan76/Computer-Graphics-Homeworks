
#include "framework.h"

vec3 rotate(vec3 p, float Z, float Y, float X) {
	mat4 rotMat = mat4(
		cos(Z) * cos(Y), cos(Z) * sin(Y) * sin(X) - sin(Z) * cos(X), cos(Z) * sin(Y) * cos(X) + sin(Z) * sin(X), 0.0,
		sin(Z) * cos(Y), cos(Z) * sin(Y) * sin(X) + cos(Z) * cos(X), sin(Z) * sin(Y) * cos(X) - cos(Z) * sin(X), 0.0,
		-sin(Y), cos(Y) * sin(X), cos(Y) * cos(X), 0.0,
		0.0, 0.0, 0.0, 1.0);
	vec4 rotated(p.x, p.y, p.z, 1);
	rotated = rotated * rotMat;
	return vec3(rotated.x, rotated.y, rotated.z);
}

vec3 shift(vec3 p, vec3 s) {
	mat4 shiftMat(
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		s.x, s.y, s.z, 1);
	vec4 shifted(p.x, p.y, p.z, 1);
	shifted = shifted * shiftMat ;
	return vec3(shifted.x, shifted.y, shifted.z);
}

struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	Material(vec3 _kd, vec3 _ks, float _shininess) : ka(_kd* M_PI), kd(_kd), ks(_ks) { shininess = _shininess; }
};

struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; }
};

vec3 operator/(vec3 num, vec3 denom) {
	return vec3(num.x / denom.x, num.y / denom.y, num.z / denom.z);
}

const vec3 one(1, 1, 1);

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

class Intersectable {
protected:
	Material* material;	
public:
	vec3 shifting;
	vec3 rot;
	vec3 translation = vec3(0, 0, 0);
	virtual Hit intersect(const Ray& ray) = 0;
	virtual Ray transform(Ray ray) = 0;
	virtual void addRot(vec3 rot) = 0;
	virtual void addShift(vec3 rota, vec3 rot) = 0;
};

// Ettõl a ponttól kezdve felhasználtam a következõ videó kódját: https://www.youtube.com/watch?v=nSHkU4fMK_g&ab_channel=LaszloSzirmay-Kalos
struct Quadrics : public Intersectable {
	mat4 Q;
	float zmin, zmax;
	vec3 translation;

	Quadrics(mat4& _Q, float _zmin, float _zmax, vec3 _trans, Material* mat, vec3 _rot) {
		Q = _Q;
		zmin = _zmin;
		zmax = _zmax;
		material = mat;
		translation = _trans;
		rot = _rot;
		shifting = (0, 0, 0);
	}

	vec3 gradf(vec3 r) {
		vec4 g = vec4(r.x, r.y, r.z, 1) * Q * 2;                                                      
		return vec3(g.x, g.y, g.z);
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 start = ray.start - translation;
		vec4 S(start.x, start.y, start.z, 1), D(ray.dir.x, ray.dir.y, ray.dir.z, 0);
		
		float a = dot(D * Q, D);
		float b = dot(S * Q, D) * 2;
		float c = dot(S * Q, S);
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;

		float sqrt_discr = sqrtf(discr);
		
		float t1 = (-b + sqrt_discr) / 2.0f / a;
		vec3 p1 = ray.start + ray.dir * t1;
		if (p1.z < zmin || p1.z > zmax) t1 = -1;

		float t2 = (-b - sqrt_discr) / 2.0f / a;
		vec3 p2 = ray.start + ray.dir * t2;
		if (p2.z < zmin || p2.z > zmax) t2 = -1;

		if (t1 <= 0 && t2 <= 0) return hit;
		if (t1 <= 0) hit.t = t2;
		else if (t2 <= 0) hit.t = t1;
		else if (t2 < t1) hit.t = t2;
		else hit.t = t1;

		hit.position = start + ray.dir + hit.t;
		hit.normal = normalize(gradf(hit.position));
		hit.position = hit.position + translation;
		hit.material = material;
		return hit;
	}

	Ray transform(Ray ray) {
		translation = translation + shifting;
		ray.dir = rotate(ray.dir, rot.z, rot.y, rot.x);
		ray.start = rotate(ray.start, rot.z, rot.y, rot.x);
		return ray;
	}

	void addRot(vec3 rota) {
		rot = rota;
	}
	void addShift(vec3 rota, vec3 rot) {
		shifting = rotate( rota, rot.z, rot.y, rot.x) ;
	}
};

const float epsilon = 0.0001f;

struct Plane : public Intersectable {
	vec3 point, normal;

	Plane(const vec3& _point, const vec3& _normal, Material* mat) {
		material = mat;
		point = _point;
		normal = _normal;
	}
	Hit intersect(const Ray& ray) {
		Hit hit;
		double NdotV = dot(normal, ray.dir);
		if (fabs(NdotV) < epsilon ) return hit;
		double t = dot(normal, point - ray.start) / NdotV;
		if (t < epsilon) return hit;
		hit.t = t;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = normal;
		if (dot(hit.normal, ray.dir) > 0) hit.normal = hit.normal * (-1); 
		hit.material = material;
		return hit;
	}

	Ray transform( Ray ray) {
		return ray;
	}

	void addRot(vec3 rota) {}
	void addShift(vec3 rota, vec3 rot) {}
};

class Camera {
	vec3 eye, lookat, right, up;
	float fov;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
		eye = _eye;
		lookat = _lookat;
		fov = _fov;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}

	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}

	void Animate(float dt) {			
		set(rotate(eye, dt,0,0), lookat, up, fov);
	}
};

struct Light {
	vec3 direction;
	vec3 Le;
	Light(vec3 _direction, vec3 _Le) {
		direction = normalize(_direction);
		Le = _Le;
	}
};

float rnd() { return (float)rand() / RAND_MAX; }

float oscY = 0.3;
float dirY = 1;

class Scene {
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	Camera camera;
	vec3 La;
public:
	void build() {
		vec3 eye = vec3(0, -15, 24), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.4f, 0.4f, 0.4f);
		vec3 lightDirection(7, 7, 20), Le(0.5, 0.5, 0.5);
		lights.push_back(new Light(lightDirection, Le));

		vec3 kd(0.3f, 0.2f, 0.1f), ks(2, 2, 2);
		vec3 kd2(0.1f, 0.2f, 0.3f);
		Material* yellow = new Material(kd, ks, 60);
		Material* blue = new Material(kd2, ks, 60);

		mat4 lampPart = ScaleMatrix(vec3(-80, -80, 0));
		objects.push_back(new Quadrics(lampPart, 0.0, 3.0, vec3(0.0, 0.0, 0.0), yellow, vec3(0.2, 0.3, 0.5)));
		objects.push_back(new Quadrics(lampPart, 3.1, 6.1, vec3(0.0, 0.0, 0.0), yellow, vec3(0.2, 0.3, 0.5)));


		mat4 lampFeet = ScaleMatrix(vec3(-1, -1, 0));
		objects.push_back(new Quadrics(lampFeet, 0.0, 0.1, vec3(0.0, 0.0, 0), yellow, vec3(0, 0, 0)));

		objects.push_back(new Plane(vec3(0, 0.0, 0), vec3(0, 0, 1), blue));

		mat4 sphere = ScaleMatrix(vec3(-70, -70, -70));
		objects.push_back(new Quadrics(sphere, -80.0, 80.0, vec3(0.0f, 0.0f, 0.05f), blue, vec3(0, 0, 0)));
		objects.push_back(new Quadrics(sphere, -80.0, 80.0, vec3(0.0f, 0.0f, 3.05f), blue, vec3(0.2, 0.3, 0.5)));
		objects.push_back(new Quadrics(sphere, -80.0, 80.0, vec3(0.0f, 0.0f, 6.15f), blue, vec3(0.2, 0.3, 0.5)));

		mat4 paraboloid = mat4(-10, 0, 0, 0,
			0, -10, 0, 0,
			0, 0, 0, 4,
			0, 0, 4, 0);
		objects.push_back(new Quadrics(paraboloid, 6.2, 7.5, vec3(0.0f, 0.0f, 6.2f), yellow, vec3(0.2, 0.3, 0.5)));
	}

	void render(std::vector<vec4>& image) {
		long timeStart = glutGet(GLUT_ELAPSED_TIME);
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
		printf("Rendering time: %d miliseconds %d", glutGet(GLUT_ELAPSED_TIME) - timeStart);
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : objects) {
			Ray newRay = object->transform(ray);
			Hit hit = object->intersect(newRay); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {	// for directional lights
		for (Intersectable* object : objects) if (object->intersect(object->transform(ray)).t > 0) return true;
		return false;
	}

	vec3 trace(Ray ray, int depth = 0) {
		
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return La;

		vec3 outRadiance = hit.material->ka * La;
		for (Light* light : lights) {
			Ray shadowRay(hit.position + hit.normal * epsilon, light->direction);
			float cosTheta = dot(hit.normal, light->direction);
			if (cosTheta > 0 && !shadowIntersect(shadowRay)) {	// shadow computation
				outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
				vec3 halfway = normalize(-ray.dir + light->direction);
				float cosDelta = dot(hit.normal, halfway);
				if (cosDelta > 0) outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
			}
		}
		return outRadiance;
	}	

	void updateRotation() {
		
		if (oscY >= 0.5) {
			dirY = -1;
		}
		
		if (oscY <= 0.0) {
			dirY = 1;
		}		
		oscY += 0.03 * dirY;

		objects[0]->addRot(vec3(objects[0]->rot.x, oscY, objects[0]->rot.z + 0.03));
		objects[5]->addRot(vec3(objects[5]->rot.x, oscY, objects[5]->rot.z + 0.03));

		objects[1]->addRot(vec3(objects[1]->rot.x, oscY, objects[1]->rot.z + 0.03));
		
		objects[6]->addRot(vec3(objects[6]->rot.x, oscY, objects[6]->rot.z + 0.03));
		objects[7]->addRot(vec3(objects[7]->rot.x, oscY, objects[7]->rot.z + 0.03));
	}

	void Animate(float dt) { 
		camera.Animate(dt); 
		updateRotation();
	}
};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
const char* vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

class FullScreenTexturedQuad {
	unsigned int vao = 0, textureId = 0;// vertex array object id and texture id

public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight) 
	{
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
	
		glGenTextures(1, &textureId);
		glBindTexture(GL_TEXTURE_2D, textureId);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);	
	}
	
	void LoadTexture(std::vector<vec4>& image) {
		glBindTexture(GL_TEXTURE_2D, textureId);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0, GL_RGBA, GL_FLOAT, &image[0]);	
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		int location = glGetUniformLocation(gpuProgram.getId(), "textureUnit");
		const unsigned int textureUnit = 0;
		if (location >= 0) {
			glUniform1i(location, textureUnit);
			glActiveTexture(GL_TEXTURE0 + textureUnit);
			glBindTexture(GL_TEXTURE_2D, textureId);
		}
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();

	std::vector<vec4> image(windowWidth * windowHeight);
	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

	// copy image to GPU as a texture
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight);

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	std::vector<vec4> image(windowWidth * windowHeight);
	scene.render(image);
	fullScreenTexturedQuad->LoadTexture(image);
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	scene.Animate(0.1f);
	glutPostRedisplay();
}
