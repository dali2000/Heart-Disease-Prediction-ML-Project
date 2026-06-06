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
            <h2 className="section-title"><Activity size={24} color="#6366f1" /> Données du Patient</h2>
            <form onSubmit={handleSubmit}>
                <div className="form-grid">
                    <div className="form-group">
                        <label>Âge (en années)</label>
                        <input type="number" name="age" min={20} max={100}
                               value={formData.age} onChange={handleChange} required />
                    </div>

                    <div className="form-group">
                        <label>Sexe</label>
                        <select name="sex" value={formData.sex} onChange={handleChange}>
                            <option value={1}>Homme</option>
                            <option value={0}>Femme</option>
                        </select>
                    </div>

                    <div className="form-group">
                        <label>Type de douleur thoracique (cp)</label>
                        <select name="cp" value={formData.cp} onChange={handleChange}>
                            <option value={0}>0 – Angine typique (douleur classique à l'effort)</option>
                            <option value={1}>1 – Angine atypique (douleur inhabituelle)</option>
                            <option value={2}>2 – Douleur non angineuse (non liée au cœur)</option>
                            <option value={3}>3 – Asymptomatique (aucune douleur)</option>
                        </select>
                    </div>

                    <div className="form-group">
                        <label>Pression artérielle au repos (mm Hg)</label>
                        <input type="number" name="trestbps" min={80} max={200}
                               value={formData.trestbps} onChange={handleChange} required />
                    </div>

                    <div className="form-group">
                        <label>Cholestérol sérique (mg/dl)</label>
                        <input type="number" name="chol" min={100} max={600}
                               value={formData.chol} onChange={handleChange} required />
                    </div>

                    <div className="form-group">
                        <label>Glycémie à jeun {'>'} 120 mg/dl (fbs)</label>
                        <select name="fbs" value={formData.fbs} onChange={handleChange}>
                            <option value={0}>Non</option>
                            <option value={1}>Oui</option>
                        </select>
                    </div>

                    <div className="form-group">
                        <label>Électrocardiogramme au repos (restecg)</label>
                        <select name="restecg" value={formData.restecg} onChange={handleChange}>
                            <option value={0}>0 – Normal</option>
                            <option value={1}>1 – Anomalie ST-T (onde T inversée ou segment ST)</option>
                            <option value={2}>2 – Hypertrophie ventriculaire gauche (cœur élargi)</option>
                        </select>
                    </div>

                    <div className="form-group">
                        <label>Fréquence cardiaque maximale atteinte (bpm)</label>
                        <input type="number" name="thalach" min={60} max={220}
                               value={formData.thalach} onChange={handleChange} required />
                    </div>

                    <div className="form-group">
                        <label>Angine induite par l'effort (exang)</label>
                        <select name="exang" value={formData.exang} onChange={handleChange}>
                            <option value={0}>Non</option>
                            <option value={1}>Oui</option>
                        </select>
                    </div>

                    <div className="form-group">
                        <label>Dépression du segment ST à l'effort (oldpeak)</label>
                        <input type="number" name="oldpeak" min={0} max={7} step={0.1}
                               value={formData.oldpeak} onChange={handleChange} required />
                    </div>

                    <div className="form-group">
                        <label>Pente du segment ST (slope)</label>
                        <select name="slope" value={formData.slope} onChange={handleChange}>
                            <option value={0}>0 – Ascendante (signe favorable)</option>
                            <option value={1}>1 – Plate (signe neutre)</option>
                            <option value={2}>2 – Descendante (signe défavorable)</option>
                        </select>
                    </div>

                    <div className="form-group">
                        <label>Nombre de vaisseaux coronaires visibles (ca, 0–3)</label>
                        <select name="ca" value={formData.ca} onChange={handleChange}>
                            <option value={0}>0</option>
                            <option value={1}>1</option>
                            <option value={2}>2</option>
                            <option value={3}>3</option>
                        </select>
                    </div>

                    <div className="form-group">
                        <label>Thalassémie – type de flux sanguin (thal)</label>
                        <select name="thal" value={formData.thal} onChange={handleChange}>
                            <option value={0}>0 – Normal</option>
                            <option value={1}>1 – Défaut fixe (zone sans flux permanent)</option>
                            <option value={2}>2 – Défaut réversible (zone sans flux à l'effort)</option>
                            <option value={3}>3 – Autre</option>
                        </select>
                    </div>

                    <button type="submit" className="btn-primary">
                        <Send size={18} /> Lancer la Prédiction
                    </button>
                </div>
            </form>
        </div>
    );
};

export default HeartForm;
