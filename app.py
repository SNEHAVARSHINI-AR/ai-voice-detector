from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import numpy as np
import tempfile
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "AI Voice Detection API is LIVE!"

@app.route('/detect-ai', methods=['POST'])
def detect_ai():
    try:
        audio_file = request.files['audio']
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            audio_file.save(tmp.name)
            
            # ============ YOUR 11-FEATURE CODE HERE ============
            y, sr = librosa.load(tmp.name, sr=None)
            
            features = {}
            features['Spectral Centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            features['Spectral Bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            features['Roll-off'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            features['Zero-Crossing Rate'] = np.mean(librosa.feature.zero_crossing_rate(y))
            features['RMS'] = np.mean(librosa.feature.rms(y=y))
            features['Tempo'] = librosa.feature.rhythm.tempo(y=y, sr=sr)[0]

            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features['MFCC'] = np.mean(np.abs(mfcc))
            
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features['Chroma'] = np.mean(chroma)
            
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch = np.mean(pitches[magnitudes > np.median(magnitudes)])
            features['Pitch'] = pitch if not np.isnan(pitch) else 0
            
            noise_level = np.std(y)
            features['Noise Level'] = noise_level

            try:
                lpc = librosa.lpc(y, order=16)
                features['LPC'] = np.mean(np.abs(lpc))
            except Exception:
                features['LPC'] = 0

            # Compare with human reference values
            human_ref = {
                'Spectral Centroid': 2000, 'Spectral Bandwidth': 2500, 'Roll-off': 4000,
                'Zero-Crossing Rate': 0.05, 'RMS': 0.07, 'Tempo': 130, 'MFCC': 7,
                'Chroma': 0.45, 'Pitch': 1000, 'Noise Level': 0.04, 'LPC': 1.5
            }

            features_list = list(features.keys())
            diff = []
            for f in features_list:
                diff_val = abs(features[f] - human_ref[f])
                if isinstance(diff_val, np.ndarray):
                    diff_val = np.mean(np.abs(diff_val))
                diff.append(diff_val)

            diff = np.array(diff, dtype=float)
            norm_diff = diff / np.max(diff)

            weights = {
                'Spectral Bandwidth': 0.2, 'Spectral Centroid': 0.1, 'Roll-off': 0.1,
                'Pitch': 0.1, 'Tempo': 0.05, 'MFCC': 0.1, 'Chroma': 0.05,
                'Zero-Crossing Rate': 0.05, 'Noise Level': 0.05, 'RMS': 0.05, 'LPC': 0.15
            }

            weighted_score = np.sum([norm_diff[i] * weights.get(features_list[i], 0.05) for i in range(len(features_list))])
            final_score = float(weighted_score)

            if final_score > 0.3:
                result = {"is_ai": True, "confidence": final_score, "message": "AI voice detected"}
            else:
                result = {"is_ai": False, "confidence": 1-final_score, "message": "Human voice detected"}
            # ============ END OF YOUR CODE ============
            
            os.unlink(tmp.name)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
