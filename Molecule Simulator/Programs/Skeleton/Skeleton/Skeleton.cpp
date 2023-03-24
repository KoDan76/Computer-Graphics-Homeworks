//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Koppa Dániel
// Neptun : G36RDE
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"
#include <time.h>

// Elõadásról másolt ImmediateModeRenderer mivel ezt egyszerûbb használni, illetve ez kell a hiperbolikus vonalak generálásához. 
// A veWo namespace-ben megtalálható a saját próbálkozásom a hiperbolikus vonalak generálására (görberajzolás nélkül, csak vektorizált szakaszokkal) 
// de ezt nem sikerült mûködésre bírnom sajnos, ezért maradtam az hiperbolikus háromszög rajzoló algoritmus osztályánál.
class ImmediateModeRenderer2D : public GPUProgram {
	const char* const vertexSource = R"(
		#version 330
		precision highp float;
		layout(location = 0) in vec2 vertexPosition;	// Attrib Array 0

		void main() { gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1); }	
	)";

	const char* const fragmentSource = R"(
		#version 330
		precision highp float;
		uniform vec3 color;
		out vec4 fragmentColor;	

		void main() { fragmentColor = vec4(color, 1); }
	)";

	unsigned int vao, vbo; // we have just a single vao and vbo for everything :-(

	int Prev(std::vector<vec2> polygon, int i) { return i > 0 ? i - 1 : polygon.size() - 1; }
	int Next(std::vector<vec2> polygon, int i) { return i < polygon.size() - 1 ? i + 1 : 0; }

	bool intersect(vec2 p1, vec2 p2, vec2 q1, vec2 q2) {
		return (dot(cross(p2 - p1, q1 - p1), cross(p2 - p1, q2 - p1)) < 0 &&
			dot(cross(q2 - q1, p1 - q1), cross(q2 - q1, p2 - q1)) < 0);
	}

	bool isEar(const std::vector<vec2>& polygon, int ear) {
		int d1 = Prev(polygon, ear), d2 = Next(polygon, ear);
		vec2 diag1 = polygon[d1], diag2 = polygon[d2];
		for (int e1 = 0; e1 < polygon.size(); e1++) { // test edges for intersection
			int e2 = Next(polygon, e1);
			vec2 edge1 = polygon[e1], edge2 = polygon[e2];
			if (d1 == e1 || d2 == e1 || d1 == e2 || d2 == e2) continue;
			if (intersect(diag1, diag2, edge1, edge2)) return false;
		}
		vec2 center = (diag1 + diag2) / 2.0f; // test middle point for being inside
		vec2 infinity(2.0f, center.y);
		int nIntersect = 0;
		for (int e1 = 0; e1 < polygon.size(); e1++) {
			int e2 = Next(polygon, e1);
			vec2 edge1 = polygon[e1], edge2 = polygon[e2];
			if (intersect(center, infinity, edge1, edge2)) nIntersect++;
		}
		return (nIntersect & 1 == 1);
	}

	void Triangulate(const std::vector<vec2>& polygon, std::vector<vec2>& triangles) {
		if (polygon.size() == 3) {
			triangles.insert(triangles.end(), polygon.begin(), polygon.begin() + 2);
			return;
		}

		std::vector<vec2> newPolygon;
		for (int i = 0; i < polygon.size(); i++) {
			if (isEar(polygon, i)) {
				triangles.push_back(polygon[Prev(polygon, i)]);
				triangles.push_back(polygon[i]);
				triangles.push_back(polygon[Next(polygon, i)]);
				newPolygon.insert(newPolygon.end(), polygon.begin() + i + 1, polygon.end());
				break;
			}
			else newPolygon.push_back(polygon[i]);
		}
		Triangulate(newPolygon, triangles); // recursive call for the rest
	}

	std::vector<vec2> Consolidate(const std::vector<vec2> polygon) {
		const float pixelThreshold = 0.01f;
		vec2 prev = polygon[0];
		std::vector<vec2> consolidatedPolygon = { prev };
		for (auto v : polygon) {
			if (length(v - prev) > pixelThreshold) {
				consolidatedPolygon.push_back(v);
				prev = v;
			}
		}
		if (consolidatedPolygon.size() > 3) {
			if (length(consolidatedPolygon.back() - consolidatedPolygon.front()) < pixelThreshold) consolidatedPolygon.pop_back();
		}
		return consolidatedPolygon;
	}

public:
	ImmediateModeRenderer2D() {
		glViewport(0, 0, windowWidth, windowHeight);
		glLineWidth(2.0f); glPointSize(10.0f);

		create(vertexSource, fragmentSource, "outColor");
		glGenVertexArrays(1, &vao); glBindVertexArray(vao);
		glGenBuffers(1, &vbo); 		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glEnableVertexAttribArray(0);  // attribute array 0
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), NULL);
	}

	void DrawGPU(int type, std::vector<vec2> vertices, vec3 color) {
		setUniform(color, "color");
		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(vec2), &vertices[0], GL_DYNAMIC_DRAW);
		glDrawArrays(type, 0, vertices.size());
	}

	void DrawPolygon(std::vector<vec2> vertices, vec3 color) {
		std::vector<vec2> triangles;
		Triangulate(Consolidate(vertices), triangles);
		DrawGPU(GL_TRIANGLE_FAN, triangles, color);
	}

	~ImmediateModeRenderer2D() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};


ImmediateModeRenderer2D* renderer; // vertex and fragment shaders
const int nTesselatedVertices = 20; // Tesszeláció mértéke a hiperbolikus szakaszokhoz
GPUProgram gpuProgram; // vertex and fragment shaders
unsigned int vao;	   // virtual world on the GPU
vec2 offset = vec2(0.1, 0.1);

 
// Elõadásról  másolt Hiperbolikus szakaszra épül 
class HyperbolicLine {
	vec2 center;
	float radius, phi_p, phi_q;
	vec2 safety1;
	vec2 safety2;
public:
	HyperbolicLine(vec2 p, vec2 q) {
		safety1 = p;
		safety2 = q;		
		float p2 = dot(p, p), q2 = dot(q, q), pq = dot(p, q);
		float a = (p2 + 1) / 2.0f, b = (q2 + 1) / 2.0f;
		float denom = (p2 * q2 - pq * pq);
		if (fabs(denom) > 1e-7) center = (p * (q2 * a - pq * b) + q * (p2 * b - pq * a)) / denom;
		vec2 center2p = p - center, center2q = q - center;
		radius = length(center2p);
		phi_p = atan2f(center2p.y, center2p.x);
		phi_q = atan2f(center2q.y, center2q.x);
		if (phi_p - phi_q >= M_PI) phi_p -= 2 * M_PI;
		else if (phi_q - phi_p >= M_PI) phi_q -= 2 * M_PI;
	}
	// Részben saját kód (Javítva, ha a szakasz átmegy a nullán, akkor egy egyenes vonalat rajzol, ahogy azt kellene neki.)
	std::vector<vec2> getTessellation() {
		std::vector<vec2> points;
		if (center.x == 0.0 && center.y == 0.0) {
			points.push_back(safety1);
			points.push_back(safety2);
		}
		else {
			points = std::vector<vec2>(nTesselatedVertices);
			for (int i = 0; i < nTesselatedVertices; i++) {
				float phi = phi_p + (phi_q - phi_p) * (float)i / (nTesselatedVertices - 1.0f);
				points[i] = center + vec2(cosf(phi), sinf(phi)) * radius;
			}
		return points;

		}
	}

	vec2 startDir(vec2 p) { return phi_q > phi_p ? normalize(center - p) : -normalize(center - p); }

	float getLength() {
		float l = -1;
		vec2 pprev;
		for (auto p : getTessellation()) {
			if (l < 0) l = 0;
			else       l += length(p - pprev) / (1 - dot((p + pprev) / 2, (p + pprev) / 2));
			pprev = p;
		}
		return l;
	}

	// Részben saját kód a kirajzoláshoz
	void Draw() {
		std::vector<vec2> polygon = getTessellation();
		renderer->DrawGPU(GL_LINE_STRIP, polygon, vec3(1.0f, 1.0f, 1.0f));
	}
};


float vec2CubicDistance(vec2 v1, vec2 v2) {
	float  arg = (v1.x - v2.x) * (v1.x - v2.x) + (v2.y - v1.y) * (v2.y - v1.y);
	return  arg;
}


vec2 hyperbolicTransform(vec2 v) {
	float w = sqrt(v.x * v.x + v.y * v.x + 1);
	return normalize(v / w);
}


namespace viWo /*as In Virtual World*/ {

	

	const float HIDROGEN_WEIGHT = 1.00784; //ku
	const float HIDROGEN_CHARGE = 1.602; // *10^-19 Coulomb
	const int aprox = 30; // Körök pontosságga 
	const float SIZE_MULT = 0.00034 * 4; // Az atomok méretének leképezése.

	class Atom {
	private:
		float charge;
		float weight;
		std::vector<Atom*> neighbors;
		vec2 coordinates;

	public:
		float I;
		vec2 v = (0.0, 0.0 );
		float om = 0.0;

		Atom(int c, float x, float y) {
			charge = HIDROGEN_CHARGE * c;
			weight = HIDROGEN_WEIGHT * (rand() % 56 + 20);
			coordinates = vec2(x, y);
			printf("Created atom with charge: %f   and weight: %f  coordinates %f %f\n", charge, weight, x, y);
		}

		vec2 getCoordinates() {
			return coordinates;
		}

		void setCoordinates(vec2 nc) {
			coordinates = nc;
		}

		void addNeighbor(Atom* n) {
			neighbors.push_back(n);
		}

		std::vector<Atom*> getneighborList() {
			return neighbors;
		}

		float getWeight() {
			return weight;
		}

		float getCharge() {
			return charge;
		}		
	};

	class Molekule {
	private:
		int numOfAtoms;
		int generation;
		int chargeRangePos;
		int chargeRangeNeg;
		int maxHeight;
		bool firstLvL = true;

		Atom* root;
		std::vector<Atom*> allAtoms; 

		void generateTree() {
			chargeRangeNeg = chargeRangePos = rand() % 1000;
			generation = numOfAtoms = rand() % 7 + 2;
			maxHeight = rand() % 3 + 2; 
			int currH = 0;
			root = generateAtom(currH, 0, 0, 40);
			normalize();
			printf("Creating molecule with % d atoms and %d height.\n Center off Mass: x:%f  y:%f\n", numOfAtoms, maxHeight, centerOfMass().x, centerOfMass().y);
		}

		
		Atom* generateAtom(int currH, float bX, float bY, int bR) {
			currH++;
			int atomCharge = 0;
			
			if (generation > 0) {
				
				if (generation > 2) {

					
					int flip = rand() % 2;
					if (flip == 1) {
						atomCharge = (rand() % chargeRangePos);
						chargeRangePos -= atomCharge;
					}
					else {
						atomCharge = -1 * (rand() % chargeRangeNeg);
						chargeRangeNeg += atomCharge;
					}
				}
				if (generation == 2) {
					atomCharge = -1 * chargeRangeNeg;
				}
				if (generation == 1) {
					atomCharge = chargeRangePos;
				}

				float nX = (rand() % bR - bR / 2 + bX) * 0.01f;

				float nY = (rand() % bR - bR / 2 + bY) * 0.01f;


				
				Atom* newAtom = new Atom(atomCharge, nX, nY);
				generation--;
				allAtoms.push_back(newAtom); 

				
				for (int i = 0; i < (firstLvL ? 7 : (rand() % 4 + 1)); i++) {
					firstLvL = true;
					if (currH != maxHeight) {
						newAtom->addNeighbor(generateAtom(currH, nY, nX, 100));
					}
				}
				
				return newAtom;
			}
			return NULL;
		}

		void normalize() {
			vec2 com = centerOfMass();
			for (auto a : allAtoms) {
				a->setCoordinates(a->getCoordinates() - com);
			}
		}

	public:
		vec2 V = (0.0,0.0);

		Molekule() { 
			generateTree();
			for (int i = 0; i < getAtoms().size(); i++) getAtoms()[i]->I = vec2CubicDistance(centerOfMass(), getAtoms()[i]->getCoordinates());
		}

		std::vector<Atom*> getAtoms() { return allAtoms; }

		vec2 centerOfMass() {
			float x = 0;
			float y = 0;
			for (auto a : allAtoms) {
				x += a->getWeight() * a->getCoordinates().x;
				y += a->getWeight() * a->getCoordinates().y;
			}
			float M = fullMass();
			return vec2(x / M, y / M);
		}

		float fullMass() {
			float M = 0;
			for (auto a : allAtoms) {
				M += a->getWeight();
			}
			return M;
		}
	};
}

// Bere Bálint által ajánlott megoldási módszer
vec2 rotateAroundPoint(vec2 point, vec2 center, float deg) {
	float s = sin(deg);
	float c = cos(deg);
	return { c * (point.x - center.x) - s * (point.y - center.y) + center.x, s * (point.x - center.x) + c * (point.y - center.y) + center.y };
}

double eps = 88541878.170; 
vec2 coloumbForce(viWo::Atom a1, viWo::Atom a2)  {
	double d = sqrt(vec2CubicDistance(a1.getCoordinates(), a2.getCoordinates()));
	
	vec2 e = vec2(a1.getCoordinates() * 300);
	e = e - a2.getCoordinates();
	e = e / d; 	
	return (a1.getCharge() * a2.getCharge()) /
				(2 * M_PI * eps * d)              * e;
}

double ro = 1000; 

//https://stackoverflow.com/questions/243945/calculating-a-2d-vectors-cross-product
double mycross(vec2 v1, vec2 v2) {
	return (v1.x * v2.y) - (v1.y * v2.x);
}



void  calculations(viWo::Molekule* mol1, viWo::Molekule* mol2) {
	const double TIMELAPSE = 0.01;
	vec2 m1Force = vec2(0.0, 0.0); 
	double M = 0.0;
	float prevRotation = 0;

	for (int i = 0; i < mol1->getAtoms().size(); i++) {

		vec2 fColoumb = vec2(0.0, 0.0); 
		mol1->getAtoms()[i]->v = mol1->V; 
		for (int j = 0; j < mol2->getAtoms().size(); j++) {  
			fColoumb = fColoumb + coloumbForce(*(mol1->getAtoms()[i]), *(mol2->getAtoms()[j]));
		}
		vec2 fRes = -ro * mol1->getAtoms()[i]->v; 
		vec2 fTotal = fRes + fColoumb; 

		M += mycross(mol1->getAtoms()[i]->getCoordinates(), fTotal);
		mol1->getAtoms()[i]->v = fTotal / mol1->fullMass() * TIMELAPSE; 
		m1Force = m1Force + fTotal;
	}
	mol1->V = mol1->V + m1Force / mol1->fullMass() * TIMELAPSE;
	vec2 com = mol1->centerOfMass();
	float molOm = 0;


	for (int i = 0; i < mol1->getAtoms().size(); i++) {
		mol1->getAtoms()[i]->v = mol1->getAtoms()[i]->v + mol1->V;
		mol1->getAtoms()[i]->om = mol1->getAtoms()[i]->om + M / mol1->getAtoms()[i]->I * TIMELAPSE;
		molOm += mol1->getAtoms()[i]->om;
		}
	molOm = molOm / mol1->getAtoms().size();
	float a = molOm * M * TIMELAPSE;

	for (int i = 0; i < mol1->getAtoms().size(); i++) {
		mol1->getAtoms()[i]->setCoordinates(rotateAroundPoint(mol1->getAtoms()[i]->getCoordinates(), com, a));
		mol1->getAtoms()[i]->setCoordinates(mol1->getAtoms()[i]->getCoordinates() + mol1->getAtoms()[i]->v);
	}
}

void simulate(viWo::Molekule* m1, viWo::Molekule* m2) {
	viWo::Molekule* mol1 = m1; 
	viWo::Molekule* mol2 = m2; 

	calculations(m1, m2);
	calculations(m2, m1);

	m1 = mol1;
	m2 = mol2;
}



class HyperbolicCircle {	
	float r;
	vec2 origo; 
	std::vector<vec2> circePoints;
	float color;

public: 
	HyperbolicCircle(float r, vec2 origo, float c) : color(c) {
		for (int i = 0; i < viWo::aprox; i++) {
			float fi = i * 2 * M_PI / viWo::aprox;
			circePoints.push_back(hyperbolicTransform(  r*vec2(cosf(fi), sinf(fi)) + origo ) );
		}
	}
	void Draw( ) {
		renderer->DrawPolygon(getPoints(),
				color >= 0 ? vec3(color / 2000 + 0.5, 0, 0) : vec3(0, 0, color / 2000 + 0.5)
			);
	}	

	std::vector<vec2> getPoints() {
		return circePoints;
	}
};


// Initialization, create an OpenGL context
void onInitialization() {
	renderer = new ImmediateModeRenderer2D();
}

viWo::Molekule* mol1;
viWo::Molekule* mol2;
std::vector<HyperbolicLine> lines;
std::vector<HyperbolicCircle> circles;

// Window has become invalid: Redraw
void onDisplay() {

	glClearColor(0.5, 0.5, 0.5, 0.5);     // background color
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear frame buffer
	srand(time(NULL));	

	if (mol1 != NULL && mol2 != NULL) {
		
		simulate( mol1, mol2);
		for (auto a : mol1->getAtoms()) {
			for (auto b: a->getneighborList()){
				if (a!= NULL && b != NULL) {
					lines.push_back(HyperbolicLine(
					 hyperbolicTransform(a->getCoordinates() + offset), hyperbolicTransform( b->getCoordinates() + offset)));
				}
				
			}
			circles.push_back( HyperbolicCircle(0.001 * a->getWeight(), 
				a->getCoordinates() + offset,
				a->getCharge())
			);
		}

		for (auto a : mol2->getAtoms()) {
			for (auto b : a->getneighborList()) {
				if (a != NULL && b != NULL) {
					lines.push_back(HyperbolicLine(
						hyperbolicTransform(a->getCoordinates() + offset ), hyperbolicTransform(b->getCoordinates() + offset )));
				}
			}
			circles.push_back(HyperbolicCircle(0.001 * a->getWeight(), a->getCoordinates() + offset, a->getCharge()));
		}	 

		
			
		for (auto l : lines) l.Draw();
		for (auto c : circles) c.Draw();
	}	

	circles.clear();
	lines.clear();		
	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	
	if (key == ' ') {
		mol1 = new viWo::Molekule();
		mol2 = new viWo::Molekule();
		offset = vec2(0.0, 0.0);
		glutPostRedisplay();
	}// if d, invalidate display, i.e. redraw
		
	switch (key)
	{
	case 'x':
		offset = offset + vec2(0.0, -0.1);
		glutPostRedisplay();
		break;
	case 's':
		offset = offset + vec2(-0.1, 0.0);
		glutPostRedisplay();
		break;
	case 'e':
		offset = offset + vec2(0.0, 0.1);
		glutPostRedisplay();
		break;
	case 'd':
		offset = offset + vec2(0.1, 0.0);
		glutPostRedisplay();
		break;
	default:
		break;
	}
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { 
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	if (time % 10 == 0 && mol1 != NULL && mol2 != NULL) {
		simulate(mol1, mol2);
		glutPostRedisplay();
	}
}
