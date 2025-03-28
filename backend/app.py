from flask import Flask, request, jsonify
from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, TIMESTAMP
from db_config import db, S3_BUCKET, S3_REGION, s3_client
from flask import request
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from torchvision import transforms
from PIL import Image
import uuid

app = Flask(__name__)

# Set up the database URL to connect to your RDS instance
app.config['SQLALCHEMY_DATABASE_URI'] = "postgresql://postgres:20Brain25?!@braindb-instance-1-us-west-1b.cb2oooakoxci.us-west-1.rds.amazonaws.com:5432/braindb"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize the database
db.init_app(app)
with app.app_context():
    db.create_all()  # This will create all the tables in the database
#classes
# Define Patient model
class Patient(db.Model):
    __tablename__ = 'patients'
    __table_args__ = {'schema': 'tumor_tracking'}
    patient_id = Column(Integer, primary_key=True)
    name = Column(String)
    date_of_birth = Column(String)
    gender = Column(String)

# Define Scan model
class Scan(db.Model):
    __tablename__ = 'scans'
    __table_args__ = {'schema': 'tumor_tracking'}
    scan_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer)
    scan_date = Column(TIMESTAMP)
    scan_url = Column(String)
    predicted_segmentation_url = Column(String)
    tumor_volume = Column(Float)
    tumor_growth_rate = Column(Float)
    tumor_type = Column(String)
    annotations = Column(JSON)

model = load_model("unet_model.h5")


# Define preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def predict_tumor(image_path):
    """Perform tumor segmentation and return mask + tumor metrics."""
    image = Image.open(image_path).convert("L")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        prediction = model(image)
        prediction = (prediction > 0.5).float()

    tumor_size = prediction.sum().item()  # Calculate tumor size in pixels
    tumor_growth_rate = 0.05  # Placeholder: Compute from previous scans

    return prediction, tumor_size, tumor_growth_rate
@app.route('/')
def index():
    return 'Welcome to the MRI Tracking API!'

@app.route('/patadd',methods =['POST'])
def add_patient():
    data = request.get_json()
    patient_id = data['patient_id']
    name = data['name']
    date_of_birth = data['dob']
    gender = data['gender']

    new_patient = Patient(patient_id, name, date_of_birth, gender)
    db.session.add(new_patient)
    db.session.commit()
    return jsonify({'message': 'Patient added successfully'}), 201

@app.route('/patients', methods=['GET'])
def get_patients():
    patients = Patient.query.all()
    return jsonify([{'patient_id': p.patient_id, 'name': p.name} for p in patients])

@app.route('/patients/<int:patient_id>', methods=['GET'])
def get_patient(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    return jsonify({'patient_id': patient.patient_id, 'name': patient.name, 'date_of_birth': str(patient.date_of_birth), 'gender': patient.gender})

@app.route('/scans', methods=['POST'])
def add_scan():
    data = request.get_json()
    patient_id = data['patient_id']
    scan_date = data['scan_date']
    scan_url = data['scan_url']
    predicted_segmentation_url = data['predicted_segmentation_url']
    tumor_volume = data['tumor_volume']
    tumor_growth_rate = data['tumor_growth_rate']
    tumor_type = data['tumor_type']
    annotations = data['annotations']

    # Add a new scan to the database
    new_scan = Scan(patient_id=patient_id, scan_date=scan_date, scan_url=scan_url,
                    predicted_segmentation_url=predicted_segmentation_url, tumor_volume=tumor_volume,
                    tumor_growth_rate=tumor_growth_rate, tumor_type=tumor_type, annotations=annotations)

    db.session.add(new_scan)
    db.session.commit()
    return jsonify({'message': 'Scan added successfully'}), 201

@app.route('/scans/<int:scan_id>', methods=['GET'])
def get_scan(scan_id):
    scan = Scan.query.get_or_404(scan_id)
    return jsonify({
        'scan_id': scan.scan_id,
        'patient_id': scan.patient_id,
        'scan_date': str(scan.scan_date),
        'scan_url': scan.scan_url,
        'predicted_segmentation_url': scan.predicted_segmentation_url,
        'tumor_volume': scan.tumor_volume,
        'tumor_growth_rate': scan.tumor_growth_rate,
        'tumor_type': scan.tumor_type,
        'annotations': scan.annotations
    })

@app.route("/upload", methods=["POST"])
def upload_scan():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    patient_id = request.form.get("patient_id")

    if not file or not patient_id:
        return jsonify({"error": "Missing file or patient_id"}), 400

    # Save the file temporarily
    filename = secure_filename(file.filename)
    temp_path = os.path.join("/tmp", filename)
    file.save(temp_path)

    # Generate unique filename for S3
    s3_filename = f"scans/{uuid.uuid4()}_{filename}"
    s3_url = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{s3_filename}"

    # Upload to S3
    s3_client.upload_file(temp_path, S3_BUCKET, s3_filename)

    # Store scan in the database
    scan = Scan(
        patient_id=patient_id,
        scan_url=s3_url,
        tumor_volume=None,  # Will be updated after prediction
        tumor_growth_rate=None,
        tumor_type=None,
        predicted_segmentation_url=None
    )
    db.session.add(scan)
    db.session.commit()

    # **Trigger tumor prediction**
    prediction, tumor_size, tumor_growth_rate = predict_tumor(temp_path)

    # Save prediction mask
    prediction_filename = f"predictions/{uuid.uuid4()}_segmentation.tif"
    prediction_path = os.path.join("/tmp", prediction_filename)
    prediction.save(prediction_path)

    # Upload prediction mask to S3
    s3_client.upload_file(prediction_path, S3_BUCKET, prediction_filename)
    prediction_url = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{prediction_filename}"

    # Update scan record with prediction results
    scan.predicted_segmentation_url = prediction_url
    scan.tumor_volume = tumor_size
    scan.tumor_growth_rate = tumor_growth_rate
    scan.tumor_type = "Glioma" if tumor_size > 5 else "Benign"  # Example rule
    db.session.commit()

    return jsonify({
        "message": "Upload successful and prediction complete",
        "scan_url": s3_url,
        "segmentation_url": prediction_url,
        "tumor_volume": tumor_size,
        "tumor_growth_rate": tumor_growth_rate
    }), 200


@app.route('/patients/<int:patient_id>/growth_rate', methods=['GET'])
def get_growth_rate(patient_id):
    # Fetch the two most recent scans for the given patient
    scans = db.session.query(Scan).filter(Scan.patient_id == patient_id).order_by(Scan.scan_date.desc()).limit(2).all()

    if len(scans) < 2:
        return jsonify({"error": "Insufficient scans to calculate growth rate."}), 400

    # Get tumor volumes and scan dates from the two most recent scans
    latest_scan = scans[0]
    previous_scan = scans[1]

    # Tumor volumes for the latest and previous scans
    latest_volume = latest_scan.tumor_volume
    previous_volume = previous_scan.tumor_volume

    # Date of the scans
    latest_scan_date = latest_scan.scan_date
    previous_scan_date = previous_scan.scan_date

    # Calculate the time difference in days
    time_difference = (latest_scan_date - previous_scan_date).days

    if time_difference == 0:
        return jsonify({"error": "The two scans are on the same day, cannot calculate growth rate."}), 400

    # Calculate the rate of growth
    growth_rate = (latest_volume - previous_volume) / time_difference

    return jsonify({
        "patient_id": patient_id,
        "growth_rate": growth_rate,
        "latest_scan_date": str(latest_scan_date),
        "previous_scan_date": str(previous_scan_date)
    })

@app.route("/scans/<int:scan_id>/annotations", methods=["PUT"])
def add_annotation(scan_id):
    """ Add or update annotations for a specific scan """
    data = request.get_json()  # Get JSON data from the request

    # Check if the scan exists
    scan = Scan.query.get(scan_id)
    if not scan:
        return jsonify({"error": "Scan not found"}), 404

    # Extract existing annotations (if any)
    existing_annotations = scan.annotations or {}

    # Merge new annotations into existing ones
    new_annotations = data.get("annotations", {})
    if not isinstance(new_annotations, dict):
        return jsonify({"error": "Annotations must be a JSON object"}), 400

    existing_annotations.update(new_annotations)
    scan.annotations = existing_annotations  # Update annotations

    # Commit to the database
    db.session.commit()

    return jsonify({"message": "Annotations updated", "annotations": scan.annotations}), 200
if __name__ == '__main__':
    app.run(debug=True)
