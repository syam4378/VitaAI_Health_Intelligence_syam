require('dotenv').config();
const FACE_URL  = process.env.FACE_SERVER_URL  || 'http://localhost:5000';
const VOICE_URL = process.env.VOICE_SERVER_URL || 'http://localhost:6000';
const express = require('express');
const mysql   = require('mysql2');
const bcrypt  = require('bcrypt');
const jwt     = require('jsonwebtoken');
const cors    = require('cors');
const path    = require('path');
const multer  = require('multer');
const fs      = require('fs');
const http    = require('http');
const fetch   = require('node-fetch');

const app = express();
app.use(express.json({ limit: '20mb' }));
app.use(cors());
app.use(express.static('public'));
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// ── MULTER ──────────────────────────────────────────────
const uploadsDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadsDir)) fs.mkdirSync(uploadsDir);
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, 'uploads/'),
  filename: (req, file, cb) => cb(null, `user_${req.user?.id||'x'}_${Date.now()}${path.extname(file.originalname)}`)
});
const upload = multer({ storage, limits: { fileSize: 5 * 1024 * 1024 } });

// ── DB ───────────────────────────────────────────────────
const db = mysql.createConnection({
  host: process.env.DB_HOST,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  database: process.env.DB_NAME,
  port: process.env.DB_PORT || 3306,   //  add this
  charset: 'utf8mb4',
  ssl: {
    rejectUnauthorized: false          // REQUIRED for Clever Cloud
  }
});
db.connect(err => {
  if (err) { console.log('DB Error:', err.message); }
  else { console.log('✅ MySQL Connected to', process.env.DB_NAME); createTables(); }
});
function q(sql, params = []) {
  return new Promise((rv, rj) => db.query(sql, params, (e, r) => e ? rj(e) : rv(r)));
}

// ── CREATE TABLES (all columns from start, no ALTER needed) ──
function createTables() {
  // users
  db.query(`CREATE TABLE IF NOT EXISTS users (
    id         INT AUTO_INCREMENT PRIMARY KEY,
    name       VARCHAR(100) NOT NULL,
    email      VARCHAR(100) UNIQUE NOT NULL,
    password   VARCHAR(255) NOT NULL,
    role       VARCHAR(20)  DEFAULT 'user',
    created_at TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
  ) CHARACTER SET utf8mb4`);

  // profile — ALL columns included from start
  db.query(`CREATE TABLE IF NOT EXISTS profile (
    id             INT AUTO_INCREMENT PRIMARY KEY,
    user_id        INT UNIQUE,
    name           VARCHAR(100),
    email          VARCHAR(100),
    phone          VARCHAR(20),
    role           VARCHAR(20),
    img            VARCHAR(255),
    age            INT,
    weight         FLOAT,
    height         FLOAT,
    blood_group    VARCHAR(5),
    gender         VARCHAR(10)  DEFAULT 'male',
    activity_level VARCHAR(30)  DEFAULT 'moderate'
  ) CHARACTER SET utf8mb4`);

  // health_checkins — ALL columns included from start
  db.query(`CREATE TABLE IF NOT EXISTS health_checkins (
    id               INT AUTO_INCREMENT PRIMARY KEY,
    user_id          INT          NOT NULL,
    sleep_quality    VARCHAR(60),
    sleep_hours      FLOAT        DEFAULT 0,
    energy_level     VARCHAR(60),
    mood             VARCHAR(60),
    water_intake     INT          DEFAULT 0,
    steps            INT          DEFAULT 0,
    calories_burned  INT          DEFAULT 0,
    exercise_done    TINYINT      DEFAULT 0,
    exercise_minutes INT          DEFAULT 0,
    exercise_type    VARCHAR(60),
    body_weight      FLOAT        DEFAULT 0,
    heart_rate       INT          DEFAULT 0,
    notes            TEXT,
    health_score     INT          DEFAULT 0,
    created_at       TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user_date (user_id, created_at)
  ) CHARACTER SET utf8mb4`);

  // health_chat_history — stores AI chat so AI remembers conversations
  db.query(`CREATE TABLE IF NOT EXISTS health_chat_history (
    id         INT AUTO_INCREMENT PRIMARY KEY,
    user_id    INT         NOT NULL,
    role       VARCHAR(20),
    content    TEXT,
    created_at TIMESTAMP   DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user (user_id)
  ) CHARACTER SET utf8mb4`);

  // Safe: add missing columns to existing tables (no error if already exist)
  const migrations = [
    "ALTER TABLE profile ADD COLUMN IF NOT EXISTS gender VARCHAR(10) DEFAULT 'male'",
    "ALTER TABLE profile ADD COLUMN IF NOT EXISTS activity_level VARCHAR(30) DEFAULT 'moderate'",
    "ALTER TABLE health_checkins ADD COLUMN IF NOT EXISTS sleep_hours FLOAT DEFAULT 0",
    "ALTER TABLE health_checkins ADD COLUMN IF NOT EXISTS steps INT DEFAULT 0",
    "ALTER TABLE health_checkins ADD COLUMN IF NOT EXISTS calories_burned INT DEFAULT 0",
    "ALTER TABLE health_checkins ADD COLUMN IF NOT EXISTS exercise_minutes INT DEFAULT 0",
    "ALTER TABLE health_checkins ADD COLUMN IF NOT EXISTS exercise_type VARCHAR(60)",
    "ALTER TABLE health_checkins ADD COLUMN IF NOT EXISTS body_weight FLOAT DEFAULT 0",
    "ALTER TABLE health_checkins ADD COLUMN IF NOT EXISTS heart_rate INT DEFAULT 0",
    "ALTER TABLE health_checkins ADD COLUMN IF NOT EXISTS health_score INT DEFAULT 0"
  ];
  migrations.forEach(sql => db.query(sql, err => {
    if (err && !err.message.includes('Duplicate column')) {
      // Ignore — column already exists
    }
  }));
  console.log('✅ Tables ready');
}

// ── MIDDLEWARE ────────────────────────────────────────────
function verifyToken(req, res, next) {
  const h = req.headers['authorization'];
  if (!h) return res.status(403).json({ message: 'No token' });
  jwt.verify(h.split(' ')[1], process.env.JWT_SECRET, (err, dec) => {
    if (err) return res.status(401).json({ message: 'Invalid token' });
    req.user = dec; next();
  });
}

// ── AI HELPER — tries Groq first, falls back to OpenRouter ──
async function callAI(messages, systemPrompt, maxTokens = 600) {
  const groqKey  = process.env.GROQ_API_KEY;
  const orKey    = process.env.OPENROUTER_API_KEY;

  // Try Groq first (fastest, free)
  if (groqKey && groqKey !== 'your_groq_api_key_here') {
    try {
      const r = await fetch('https://api.groq.com/openai/v1/chat/completions', {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${groqKey}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'llama-3.3-70b-versatile',
          messages: [{ role: 'system', content: systemPrompt }, ...messages],
          temperature: 0.7, max_tokens: maxTokens
        })
      });
      const data = await r.json();
      if (!data.error) return data.choices[0].message.content;
      console.log('Groq error, trying OpenRouter:', data.error.message);
    } catch(e) { console.log('Groq failed, trying OpenRouter:', e.message); }
  }

  // Fallback to OpenRouter
  if (orKey) {
    const r = await fetch('https://openrouter.ai/api/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${orKey}`,
        'Content-Type': 'application/json',
        'HTTP-Referer': 'http://localhost:3000',
        'X-Title': 'VitaAI Health Platform'
      },
      body: JSON.stringify({
        model: 'meta-llama/llama-3.1-8b-instruct:free',
        messages: [{ role: 'system', content: systemPrompt }, ...messages],
        temperature: 0.7, max_tokens: maxTokens
      })
    });
    const data = await r.json();
    if (data.error) throw new Error(data.error.message || 'OpenRouter error');
    return data.choices[0].message.content;
  }

  throw new Error('No AI API key configured. Add GROQ_API_KEY to .env');
}

// ── PYTHON PROXY ──────────────────────────────────────────
function callPython(route, body) {
  return new Promise((rv, rj) => {
    const data = JSON.stringify(body);
    const req  = http.request({
      hostname: new URL(FACE_URL).hostname,
      port: new URL(FACE_URL).port || 80, path: route, method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(data) }
    }, res => {
      let raw = '';
      res.on('data', c => raw += c);
      res.on('end', () => { try { rv(JSON.parse(raw)); } catch(e) { rj(new Error('Bad face server response')); } });
    });
    req.on('error', () => rj(new Error('Face server not running. Start Face_server.py first.')));
    req.write(data); req.end();
  });
}

// ── PAGES ─────────────────────────────────────────────────
app.get('/',       (req, res) => res.sendFile(path.join(__dirname, 'public', 'signup.html')));
app.get('/home',   (req, res) => res.sendFile(path.join(__dirname, 'public', 'home.html')));
app.get('/login',  (req, res) => res.sendFile(path.join(__dirname, 'public', 'login.html')));
app.get('/signup', (req, res) => res.sendFile(path.join(__dirname, 'public', 'signup.html')));

// ── FACE ──────────────────────────────────────────────────
app.post('/face/register', async (req, res) => {
  try { res.json(await callPython('/face/register', req.body)); }
  catch(e) { res.status(500).json({ ok: false, msg: e.message }); }
});
app.post('/face/verify', async (req, res) => {
  try { res.json(await callPython('/face/verify', req.body)); }
  catch(e) { res.status(500).json({ ok: false, msg: e.message }); }
});
// Emotion detection endpoint for mood scanner
app.post('/face/emotion', async (req, res) => {
  try { res.json(await callPython('/face/emotion', req.body)); }
  catch(e) { res.status(200).json({ emotion: 'neutral', ok: false }); } // fail gracefully
});

// ── VOICE ─────────────────────────────────────────────────
app.post('/voice/chat', verifyToken, (req, res) => {
  const chunks = [];
  req.on('data', c => chunks.push(c));
  req.on('end', () => {
    const audio = Buffer.concat(chunks);
    const pr = http.request({
      hostname: new URL(VOICE_URL).hostname,
      port: new URL(VOICE_URL).port || 80, path: '/voice/chat', method: 'POST',
      headers: { 'Content-Type': req.headers['content-type'] || 'audio/webm', 'Content-Length': audio.length }
    }, proxyRes => {
      res.set({
        'Content-Type': 'audio/wav', 'Cache-Control': 'no-cache',
        'X-User-Text': proxyRes.headers['x-user-text'] || '',
        'X-Agent-Text': proxyRes.headers['x-agent-text'] || '',
        'Access-Control-Expose-Headers': 'X-User-Text, X-Agent-Text'
      });
      proxyRes.pipe(res);
    });
    pr.on('error', () => res.status(500).json({ error: 'Voice agent offline. Start voice_agent.py' }));
    pr.write(audio); pr.end();
  });
});

// ── AUTH ──────────────────────────────────────────────────
app.post('/signup', async (req, res) => {
  const { name, email, password, role } = req.body;
  if (!name || !email || !password) return res.json({ message: 'All fields required' });
  const hash = await bcrypt.hash(password, 10);
  db.query('INSERT INTO users (name,email,password,role) VALUES (?,?,?,?)',
    [name, email, hash, role || 'user'], (err, result) => {
      if (err) return res.json({ message: err.code === 'ER_DUP_ENTRY' ? 'Email already registered' : 'Signup error: ' + err.message });
      db.query('INSERT INTO profile (user_id,name,email,role) VALUES (?,?,?,?)',
        [result.insertId, name, email, role || 'user'], () => {});
      res.json({ message: 'Signup successful' });
    });
});

app.post('/login', (req, res) => {
  const { email, password } = req.body;
  db.query('SELECT * FROM users WHERE email=?', [email], async (err, r) => {
    if (err || !r.length) return res.json({ message: 'User not found' });
    if (!await bcrypt.compare(password, r[0].password)) return res.json({ message: 'Wrong password' });
    res.json({ token: jwt.sign({ id: r[0].id, name: r[0].name, email: r[0].email, role: r[0].role }, process.env.JWT_SECRET, { expiresIn: '24h' }) });
  });
});

app.post('/face-login', (req, res) => {
  db.query('SELECT * FROM users WHERE email=?', [req.body.email], (err, r) => {
    if (err || !r.length) return res.json({ message: 'User not found' });
    res.json({ token: jwt.sign({ id: r[0].id, name: r[0].name, email: r[0].email, role: r[0].role }, process.env.JWT_SECRET, { expiresIn: '24h' }) });
  });
});

// ── PROFILE ───────────────────────────────────────────────
app.get('/profile', verifyToken, (req, res) => {
  db.query('SELECT * FROM profile WHERE user_id=?', [req.user.id], (err, r) => {
    if (err) return res.status(500).json({ message: 'DB error: ' + err.message });
    res.json({ user: r[0] || null });
  });
});

// Profile save — handles multipart with optional image
app.post('/profile', verifyToken, (req, res) => {
  upload.single('img')(req, res, (multerErr) => {
    const { name, email, phone, age, weight, height, blood_group, gender, activity_level } = req.body;
    const newImg = req.file ? req.file.filename : null;

    db.query('SELECT id, img FROM profile WHERE user_id=?', [req.user.id], (err, r) => {
      if (err) return res.status(500).json({ message: 'DB error: ' + err.message });

      const oldImg = r[0]?.img || null;
      const imgToSave = newImg || oldImg;

      if (newImg && oldImg) {
        const op = path.join(__dirname, 'uploads', oldImg);
        if (fs.existsSync(op)) try { fs.unlinkSync(op); } catch(e) {}
      }

      const vals = [
        name || '', email || '', phone || '', imgToSave,
        parseInt(age) || null, parseFloat(weight) || null,
        parseInt(height) || null, blood_group || '',
        gender || 'male', activity_level || 'moderate'
      ];

      if (r.length > 0) {
        db.query(
          'UPDATE profile SET name=?,email=?,phone=?,img=?,age=?,weight=?,height=?,blood_group=?,gender=?,activity_level=? WHERE user_id=?',
          [...vals, req.user.id],
          e2 => {
            if (e2) return res.status(500).json({ message: 'Update error: ' + e2.message });
            res.json({ message: 'Profile updated', img: imgToSave });
          }
        );
      } else {
        db.query(
          'INSERT INTO profile (user_id,name,email,phone,img,age,weight,height,blood_group,gender,activity_level) VALUES (?,?,?,?,?,?,?,?,?,?,?)',
          [req.user.id, ...vals],
          e2 => {
            if (e2) return res.status(500).json({ message: 'Insert error: ' + e2.message });
            res.json({ message: 'Profile saved', img: imgToSave });
          }
        );
      }
    });
  });
});

// ── HEALTH SCORE CALCULATOR ───────────────────────────────
function calcScore(c) {
  let score = 0, maxPts = 0;

  // Sleep — 25 pts
  const sh = parseFloat(c.sleep_hours) || 0;
  if (sh > 0) {
    maxPts += 25;
    if (sh >= 7 && sh <= 9) score += 25;
    else if (sh >= 6)  score += 18;
    else if (sh >= 5)  score += 10;
    else               score += 4;
  } else if (c.sleep_quality) {
    maxPts += 25;
    const sq = c.sleep_quality;
    if (sq.includes('8+') || sq.includes('7-8') || sq.includes('7–8')) score += 25;
    else if (sq.includes('5-6') || sq.includes('5–6'))                  score += 14;
    else                                                                  score += 5;
  }

  // Steps — 20 pts
  const st = parseInt(c.steps) || 0;
  if (st > 0) {
    maxPts += 20;
    if (st >= 10000) score += 20;
    else if (st >= 7500) score += 16;
    else if (st >= 5000) score += 12;
    else if (st >= 3000) score += 8;
    else                 score += 4;
  }

  // Water — 15 pts
  const w = parseInt(c.water_intake) || 0;
  if (w > 0) {
    maxPts += 15;
    if (w >= 8) score += 15;
    else if (w >= 6) score += 12;
    else if (w >= 4) score += 8;
    else             score += 4;
  }

  // Mood — 20 pts (always counted if provided)
  if (c.mood) {
    maxPts += 20;
    const m = c.mood;
    if (m.includes('Great!')) score += 20;
    else if (m.includes('Good')) score += 15;
    else if (m.includes('Neutral')) score += 10;
    else score += 4;
  }

  // Energy — 10 pts (always counted if provided)
  if (c.energy_level) {
    maxPts += 10;
    const e = c.energy_level;
    if (e.includes('High')) score += 10;
    else if (e.includes('Medium')) score += 8;
    else if (e.includes('Low') && !e.includes('Very')) score += 5;
    else score += 2;
  }

  // Exercise — 10 pts
  const em = parseInt(c.exercise_minutes) || 0;
  if (em > 0 || c.exercise_done) {
    maxPts += 10;
    if (em >= 45) score += 10;
    else if (em >= 30) score += 8;
    else if (em >= 15) score += 5;
    else score += 3;
  }

  if (maxPts === 0) return 0;
  // Scale to 100
  return Math.min(100, Math.round((score / maxPts) * 100));
}

// BMR Mifflin-St Jeor
function calcCalGoal(p) {
  if (!p || !p.weight || !p.height || !p.age) return 2000;
  const bmr = p.gender === 'female'
    ? 447.593 + (9.247 * p.weight) + (3.098 * p.height) - (4.330 * p.age)
    : 88.362 + (13.397 * p.weight) + (4.799 * p.height) - (5.677 * p.age);
  const m = { sedentary: 1.2, light: 1.375, moderate: 1.55, active: 1.725, very_active: 1.9 };
  return Math.round(bmr * (m[p.activity_level] || 1.55));
}
function stepsToCalories(steps, weight = 70) {
  return Math.round(steps * 0.04 * (weight / 70));
}

// ── CHECKIN SAVE ──────────────────────────────────────────
app.post('/checkin', verifyToken, async (req, res) => {
  const {
    sleep_quality, sleep_hours, energy_level, mood,
    water_intake, steps, calories_burned, exercise_done,
    exercise_minutes, exercise_type, body_weight, heart_rate, notes
  } = req.body;

  let prof = null;
  try { const r = await q('SELECT * FROM profile WHERE user_id=?', [req.user.id]); prof = r[0] || null; } catch(e) {}

  const stepsN    = parseInt(steps) || 0;
  const calIn     = parseInt(calories_burned) || 0;
  const finalCal  = calIn > 0 ? calIn : (stepsN > 0 ? stepsToCalories(stepsN, prof?.weight || 70) : 0);
  const hs = calcScore({ sleep_quality, sleep_hours, energy_level, mood, water_intake, steps: stepsN, exercise_done, exercise_minutes });

  try {
    await q(
      `INSERT INTO health_checkins
        (user_id,sleep_quality,sleep_hours,energy_level,mood,water_intake,steps,
         calories_burned,exercise_done,exercise_minutes,exercise_type,
         body_weight,heart_rate,notes,health_score)
       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`,
      [req.user.id,
       sleep_quality || '', parseFloat(sleep_hours) || 0,
       energy_level || '', mood || '',
       parseInt(water_intake) || 0, stepsN, finalCal,
       exercise_done ? 1 : 0, parseInt(exercise_minutes) || 0,
       exercise_type || null, parseFloat(body_weight) || 0,
       parseInt(heart_rate) || 0, notes || '', hs]
    );
    // Save to chat history for AI memory
    const summary = `Check-in: sleep=${sleep_quality}(${sleep_hours||0}h), energy=${energy_level}, mood=${mood}, steps=${stepsN}, water=${water_intake||0}, cal=${finalCal}, score=${hs}/100`;
    db.query('INSERT INTO health_chat_history (user_id,role,content) VALUES (?,?,?)',
      [req.user.id, 'system', summary], () => {});

    res.json({ message: 'Check-in saved!', health_score: hs, calories_auto: finalCal });
  } catch(e) {
    res.status(500).json({ message: 'Check-in failed: ' + e.message });
  }
});

app.get('/checkins', verifyToken, (req, res) => {
  db.query('SELECT * FROM health_checkins WHERE user_id=? ORDER BY created_at DESC LIMIT 30',
    [req.user.id], (err, r) => err ? res.status(500).json({ message: 'Error' }) : res.json({ checkins: r }));
});

// ── DASHBOARD STATS ───────────────────────────────────────
app.get('/dashboard/stats', verifyToken, async (req, res) => {
  const uid = req.user.id;
  try {
    const profRows = await q('SELECT * FROM profile WHERE user_id=?', [uid]);
    const prof     = profRows[0] || null;
    const checkins = await q('SELECT * FROM health_checkins WHERE user_id=? ORDER BY created_at DESC LIMIT 30', [uid]);
    const todayRows = await q(
      'SELECT * FROM health_checkins WHERE user_id=? AND DATE(created_at)=CURDATE() ORDER BY created_at DESC LIMIT 1', [uid]);
    const today = todayRows[0] || null;
    const total = checkins.length;

    // Avg score from last 7 valid checkins
    const last7    = checkins.slice(0, 7);
    const withScore = last7.filter(c => (c.health_score || 0) > 0);
    const avgScore  = withScore.length
      ? Math.round(withScore.reduce((s, c) => s + c.health_score, 0) / withScore.length)
      : 0;

    // Streak — consecutive days with check-in
    let streak = 0;
    const dates = checkins.map(c => new Date(c.created_at).toDateString());
    let chk = new Date();
    while (dates.includes(chk.toDateString())) { streak++; chk.setDate(chk.getDate() - 1); }

    const calorieGoal = calcCalGoal(prof);

    let bmi = null, bmiLabel = '';
    if (prof?.weight && prof?.height) {
      const h = prof.height / 100;
      bmi = (prof.weight / (h * h)).toFixed(1);
      const b = parseFloat(bmi);
      bmiLabel = b < 18.5 ? 'Underweight' : b < 25 ? 'Normal ✅' : b < 30 ? 'Overweight' : 'Obese';
    }

    // Charts — 7 days for first week, 30 days after
    const chartData = checkins.slice(0, total >= 14 ? 30 : 7).reverse();
    const days      = chartData.map(c => new Date(c.created_at).toLocaleDateString('en-IN', { month: 'short', day: 'numeric' }));
    const scoreData = chartData.map(c => c.health_score || 0);
    const stepsData = chartData.map(c => c.steps || 0);
    const calData   = chartData.map(c => c.calories_burned || 0);
    const waterData = chartData.map(c => c.water_intake || 0);
    const sleepData = chartData.map(c => {
      if (c.sleep_hours > 0) return parseFloat(c.sleep_hours);
      const sq = c.sleep_quality || '';
      if (sq.includes('8+'))       return 8.5;
      if (sq.includes('7-8') || sq.includes('7–8')) return 7.5;
      if (sq.includes('5-6') || sq.includes('5–6')) return 5.5;
      if (sq.includes('5h'))       return 4;
      return null;
    });
    const moodScore = { 'Great!': 5, 'Good': 4, 'Neutral': 3, 'Sad/Anxious': 1 };
    const moodData = chartData.map(c => {
      for (const [k, v] of Object.entries(moodScore)) { if ((c.mood || '').includes(k)) return v; }
      return null;
    });

    res.json({
      hasData: total > 0, score: avgScore, totalCheckins: total, streak,
      calorieGoal, bmi, bmiLabel,
      profile: prof ? { age: prof.age, weight: prof.weight, height: prof.height, blood_group: prof.blood_group, gender: prof.gender, activity_level: prof.activity_level } : null,
      today: today ? {
        sleep: today.sleep_quality, sleep_hours: today.sleep_hours,
        energy: today.energy_level, mood: today.mood,
        water: today.water_intake, steps: today.steps,
        calories: today.calories_burned, exercise_done: today.exercise_done,
        exercise_minutes: today.exercise_minutes, exercise_type: today.exercise_type,
        body_weight: today.body_weight, heart_rate: today.heart_rate,
        health_score: today.health_score
      } : null,
      charts: { days, scoreData, moodData, sleepData, stepsData, calData, waterData },
      chartDays: total >= 14 ? 30 : 7
    });
  } catch(e) { res.status(500).json({ error: e.message }); }
});

// ── AI CHAT — Groq + OpenRouter fallback ─────────────────
app.post('/ai/chat', verifyToken, async (req, res) => {
  const { messages } = req.body;

  let ctx = '';
  try {
    const pr = await q('SELECT * FROM profile WHERE user_id=?', [req.user.id]);
    const ci = await q('SELECT * FROM health_checkins WHERE user_id=? ORDER BY created_at DESC LIMIT 3', [req.user.id]);
    const p  = pr[0];
    if (p) {
      ctx = `User: ${req.user.name}`;
      if (p.age)    ctx += `, Age: ${p.age}`;
      if (p.weight) ctx += `, Weight: ${p.weight}kg`;
      if (p.height) ctx += `, Height: ${p.height}cm`;
      if (p.blood_group) ctx += `, Blood: ${p.blood_group}`;
      if (p.weight && p.height) { const h = p.height/100; ctx += `, BMI: ${(p.weight/(h*h)).toFixed(1)}`; }
      ctx += `, Gender: ${p.gender || 'male'}`;
    }
    if (ci.length > 0) {
      const l = ci[0];
      ctx += `\nLatest check-in: Sleep ${l.sleep_hours > 0 ? l.sleep_hours+'h' : l.sleep_quality}, Energy: ${l.energy_level}, Mood: ${l.mood}, Steps: ${(l.steps||0).toLocaleString()}, Water: ${l.water_intake||0} glasses, Cal burned: ${l.calories_burned||0} kcal, HR: ${l.heart_rate||'—'}, Health score: ${l.health_score||0}/100`;
    }
  } catch(e) {}

  const sysPrompt = `You are SyamHealth AI, a warm personal health coach.\n${ctx ? 'User Profile:\n' + ctx : ''}\nGive friendly, actionable, personalized health advice in 2-4 sentences. NEVER give dangerous advice. For serious symptoms say "please consult a doctor". Use emojis occasionally.`;

  try {
    const reply = await callAI(messages.slice(-10), sysPrompt, 600);
    // Save to chat history
    db.query('INSERT INTO health_chat_history (user_id,role,content) VALUES (?,?,?)',
      [req.user.id, 'assistant', reply], () => {});
    res.json({ reply });
  } catch(e) { res.status(500).json({ error: 'AI error: ' + e.message }); }
});

// ── DAILY HEALTH REPORT ───────────────────────────────────
app.get('/ai/daily-report', verifyToken, async (req, res) => {
  try {
    const profRows = await q('SELECT * FROM profile WHERE user_id=?', [req.user.id]);
    const ci       = await q('SELECT * FROM health_checkins WHERE user_id=? ORDER BY created_at DESC LIMIT 7', [req.user.id]);
    const prof     = profRows[0] || null;
    const today    = ci[0] || null;

    if (!today) return res.json({ report: 'No check-in data yet. Complete a Daily Check-in first to get your personalized health report! 📋\n\nOnce you do your first check-in, I will generate:\n• Summary of your health today\n• What you did well\n• What needs improvement\n• Specific food and exercise recommendations\n• Goals for tomorrow' });

    const avg7Steps = ci.length > 0 ? Math.round(ci.reduce((s, c) => s + (c.steps || 0), 0) / ci.length) : 0;
    const avg7Score = ci.filter(c => c.health_score > 0).length > 0
      ? Math.round(ci.filter(c => c.health_score > 0).reduce((s, c) => s + c.health_score, 0) / ci.filter(c => c.health_score > 0).length)
      : 0;

    const prompt = `Generate a detailed daily health report for this user.

${prof?.name ? `Name: ${prof.name}` : ''}
${prof?.age ? `Age: ${prof.age}, Gender: ${prof.gender || 'male'}, Weight: ${prof.weight}kg, Height: ${prof.height}cm` : ''}

TODAY'S CHECK-IN DATA:
- Sleep: ${today.sleep_quality} (${today.sleep_hours}h)
- Energy: ${today.energy_level}
- Mood: ${today.mood}
- Steps walked: ${(today.steps || 0).toLocaleString()}
- Water: ${today.water_intake || 0} glasses
- Calories burned: ${today.calories_burned || 0} kcal
- Exercise: ${today.exercise_minutes || 0} minutes ${today.exercise_type || ''}
- Heart Rate: ${today.heart_rate > 0 ? today.heart_rate + ' bpm' : 'not logged'}
- Health Score: ${today.health_score || 0}/100

7-DAY AVERAGES:
- Avg health score: ${avg7Score}/100
- Avg daily steps: ${avg7Steps.toLocaleString()}

Please provide in this format:
📊 **Today's Health Summary**
(2-3 sentences about today)

✅ **What You Did Well**
(2-3 specific things)

⚠️ **What Needs Improvement**
(2-3 specific things with reasons)

🥗 **Food Recommendations for Tomorrow**
(5 specific foods with reasons)

💪 **Exercise Plan for Tomorrow**
(specific exercise with duration)

😴 **Sleep Tip**
(one specific tip)

🎯 **Tomorrow's Goal**
(one clear measurable goal)

Be warm, specific, and motivating. Use the actual numbers from their data.`;

    const report = await callAI([{ role: 'user', content: prompt }], 'You are a personal health coach generating a daily health report. Be specific, warm, and use the actual data provided.', 800);
    res.json({ report });
  } catch(e) { res.status(500).json({ error: 'Report error: ' + e.message }); }
});

// ── ANALYZE ANY HEALTH REPORT ─────────────────────────────
app.post('/ai/analyze-report', verifyToken, async (req, res) => {
  const { reportText } = req.body;
  const sysPrompt = `You are a medical report analyzer. Analyze ANY type of health/medical report (blood, urine, thyroid, diabetes, lipid panel, liver, kidney, CBC, etc.).
Extract ALL test values and return ONLY a JSON array.
Format: [{"name":"Test Name","value":"result","unit":"unit","status":"normal|high|low","range":"normal range","advice":"specific actionable advice in 1-2 sentences — include food, exercise, or medical action"}]
For each HIGH or LOW value, give specific advice like: "Eat iron-rich foods (spinach, lentils, dates), avoid tea/coffee after meals" or "Reduce saturated fat, exercise 30min/day".
Return ONLY valid JSON array. No text, no markdown backticks.`;
  try {
    let txt = await callAI([{ role: 'user', content: `Analyze this health report:\n${(reportText || '').slice(0, 4000)}` }], sysPrompt, 1500);
    txt = txt.trim().replace(/```json|```/g, '').trim();
    try { res.json({ items: JSON.parse(txt) }); }
    catch(e) { res.json({ items: [], raw: txt }); }
  } catch(e) { res.status(500).json({ error: e.message }); }
});

// ── CHAT HISTORY (what health_chat_history stores) ────────
// It stores: AI conversations + check-in summaries so AI remembers user context
app.get('/chat-history', verifyToken, (req, res) => {
  db.query('SELECT role, content, created_at FROM health_chat_history WHERE user_id=? ORDER BY created_at DESC LIMIT 20',
    [req.user.id], (err, r) => err ? res.status(500).json({ error: err.message }) : res.json({ history: r }));
});

app.get('/config/chatbase', (req, res) => res.json({ botId: process.env.CHATBASE_BOT_ID || '' }));

app.listen(process.env.PORT || 3000, () => {
  console.log(`\n🚀 VitaAI Server: http://localhost:${process.env.PORT || 3000}`);
  console.log(`   Groq AI: ${process.env.GROQ_API_KEY ? '✅' : '❌ missing'}`);
  console.log(`   OpenRouter: ${process.env.OPENROUTER_API_KEY ? '✅' : '❌ missing'}\n`);
});