import React, { useState } from 'react';
import { Send, Activity } from 'lucide-react';

const HeartForm = ({ onPredict }) => {
    const [formData, setFormData] = useState({
        age: 55,
        sex: 1,
        cp: 0,
        trestbps: 130,
        chol: 250,
        fbs: 0,
        restecg: 0,
        thalach: 150,
        exang: 0,
        oldpeak: 1.0,
        slope: 1,
        ca: 0,
        thal: 2
    });

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData(prev => ({
            ...prev,
            [name]: parseFloat(value)
        }));
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        onPredict(formData);
    };

    return (
        <div className="glass-card">
            <h2 className="section-title"><Activity size={24} color="#6366f1" /> Patient Data</h2>
            <form onSubmit={handleSubmit}>
                <div className="form-grid">
                    <div className="form-group">
                        <label>Age (years)</label>
                        <input type="number" name="age" min={20} max={100}
                               value={formData.age} onChange={handleChange} required />
                    </div>

                    <div className="form-group">
                        <label>Sex</label>
                        <select name="sex" value={formData.sex} onChange={handleChange}>
                            <option value={1}>Male (1)</option>
                            <option value={0}>Female (0)</option>
                        </select>
                    </div>

                    <div className="form-group">
                        <label>Chest Pain Type (cp)</label>
                        <select name="cp" value={formData.cp} onChange={handleChange}>
                            <option value={0}>0 – Typical Angina</option>
                            <option value={1}>1 – Atypical Angina</option>
                            <option value={2}>2 – Non-Anginal Pain</option>
                            <option value={3}>3 – Asymptomatic</option>
                        </select>
                    </div>

                    <div className="form-group">
                        <label>Resting BP (trestbps, mm Hg)</label>
                        <input type="number" name="trestbps" min={80} max={200}
                               value={formData.trestbps} onChange={handleChange} required />
                    </div>

                    <div className="form-group">
                        <label>Cholesterol (chol, mg/dl)</label>
                        <input type="number" name="chol" min={100} max={600}
                               value={formData.chol} onChange={handleChange} required />
                    </div>

                    <div className="form-group">
                        <label>Fasting Blood Sugar {'>'} 120 (fbs)</label>
                        <select name="fbs" value={formData.fbs} onChange={handleChange}>
                            <option value={0}>No (0)</option>
                            <option value={1}>Yes (1)</option>
                        </select>
                    </div>

                    <div className="form-group">
                        <label>Resting ECG (restecg)</label>
                        <select name="restecg" value={formData.restecg} onChange={handleChange}>
                            <option value={0}>0 – Normal</option>
                            <option value={1}>1 – ST-T Abnormality</option>
                            <option value={2}>2 – LV Hypertrophy</option>
                        </select>
                    </div>

                    <div className="form-group">
                        <label>Max Heart Rate (thalach)</label>
                        <input type="number" name="thalach" min={60} max={220}
                               value={formData.thalach} onChange={handleChange} required />
                    </div>

                    <div className="form-group">
                        <label>Exercise Angina (exang)</label>
                        <select name="exang" value={formData.exang} onChange={handleChange}>
                            <option value={0}>No (0)</option>
                            <option value={1}>Yes (1)</option>
                        </select>
                    </div>

                    <div className="form-group">
                        <label>ST Depression (oldpeak)</label>
                        <input type="number" name="oldpeak" min={0} max={7} step={0.1}
                               value={formData.oldpeak} onChange={handleChange} required />
                    </div>

                    <div className="form-group">
                        <label>ST Slope (slope)</label>
                        <select name="slope" value={formData.slope} onChange={handleChange}>
                            <option value={0}>0 – Upsloping</option>
                            <option value={1}>1 – Flat</option>
                            <option value={2}>2 – Downsloping</option>
                        </select>
                    </div>

                    <div className="form-group">
                        <label>Major Vessels (ca, 0-3)</label>
                        <select name="ca" value={formData.ca} onChange={handleChange}>
                            <option value={0}>0</option>
                            <option value={1}>1</option>
                            <option value={2}>2</option>
                            <option value={3}>3</option>
                        </select>
                    </div>

                    <div className="form-group">
                        <label>Thalassemia (thal)</label>
                        <select name="thal" value={formData.thal} onChange={handleChange}>
                            <option value={0}>0 – Normal</option>
                            <option value={1}>1 – Fixed Defect</option>
                            <option value={2}>2 – Reversable Defect</option>
                            <option value={3}>3 – Other</option>
                        </select>
                    </div>

                    <button type="submit" className="btn-primary">
                        <Send size={18} /> Run Prediction
                    </button>
                </div>
            </form>
        </div>
    );
};

export default HeartForm;
