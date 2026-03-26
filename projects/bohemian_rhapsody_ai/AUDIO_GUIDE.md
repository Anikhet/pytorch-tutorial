# Audio Guide 🎸🔊

The Guitar Learning AI now includes realistic guitar-like audio synthesis!

## How It Works

The audio system generates guitar tones with:
- **Harmonics**: Fundamental frequency + overtones (2nd, 3rd, 4th, 5th harmonics)
- **ADSR Envelope**: Attack, Decay, Sustain, Release for realistic note shaping
- **Chorus Effect**: Slight detuning for richer sound
- **Smart Caching**: Pre-generates common notes for low latency

## Using Audio

### Demo Mode (Audio ON by default)

```bash
python3 demo.py
```

**Controls:**
- `M`: Toggle audio on/off
- Audio indicator shows in bottom-right corner (🔊 = on, 🔇 = off)

You'll hear the bot play each note as it attempts them!

### Training Mode (Audio OFF by default)

For training (`main.py` and `main_visual.py`), audio is **disabled by default** to avoid overwhelming sound from 8 players learning simultaneously.

If you want to enable audio during training, you can modify the code:

**In `main.py` or `main_visual.py`**, change:
```python
players = [GuitarPlayer(network) for network in ga.get_population()]
```

To:
```python
players = [GuitarPlayer(network, audio_enabled=True) for network in ga.get_population()]
```

**Warning**: With 8 players all playing random notes early in training, it will be VERY noisy!

## Audio Quality

The synthesizer produces:
- **Sample Rate**: 22050 Hz
- **Bit Depth**: 16-bit
- **Channels**: Stereo
- **Format**: Guitar-like tones with harmonics

### What You'll Hear

**Early Training (Generations 0-10)**:
- Random notes
- Chaotic timing
- Many wrong pitches

**Mid Training (Generations 10-30)**:
- Some correct notes emerging
- Better timing
- Recognizable patterns

**Late Training (Generations 30+)**:
- Clear melodic phrases
- Accurate timing
- Smooth performance

## Adjusting Volume

To change volume, edit `audio_synth.py`:

```python
self.volume = 0.3  # Default (0.0 to 1.0)
```

Or programmatically:
```python
import audio_synth
audio_synth.set_volume(0.5)  # 50% volume
```

## Troubleshooting

### "Audio synthesizer not initialized"

Try reinstalling pygame:
```bash
pip3 install --upgrade pygame
```

### No sound is playing

1. Check your system volume
2. Make sure audio is enabled (press `M` in demo)
3. Verify pygame mixer is working:
   ```bash
   python3 audio_synth.py
   ```

### Crackling or popping sounds

This can happen if:
- CPU is overloaded (try closing other apps)
- Buffer size is too small (increase in `audio_synth.py`)
- Too many sounds playing at once (normal during training with 8 players)

### Audio lag

The system pre-caches common notes to minimize latency. If you notice lag:
- First run takes longer (caching sounds)
- Subsequent runs are faster
- Demo mode has better performance than training

## Technical Details

### Harmonics Used

1. **Fundamental** (1.0x) - Main frequency
2. **2nd harmonic** (0.5x) - Octave
3. **3rd harmonic** (0.3x) - Fifth above octave
4. **4th harmonic** (0.15x) - Two octaves
5. **5th harmonic** (0.1x) - Major third
6. **Detuned** (0.2x at 1.01x) - Chorus effect

### ADSR Envelope

- **Attack**: 10ms (fast pick attack)
- **Decay**: 50ms (initial brightness fade)
- **Sustain**: 70% level (held note)
- **Release**: 100ms (note fade out)

### Cache Performance

- Pre-caches all solo notes at startup
- Supports up to 200 cached sounds
- Cache key: `(frequency * 10, duration * 100)`

## Comparison: Silent vs Audio Training

### Silent Training (Recommended)
✅ Faster (no audio overhead)
✅ Cleaner (no noise)
✅ Better for long training sessions
✅ Default setting

### Audio Training
✅ More engaging to watch
✅ Hear learning progress
✅ Entertaining
❌ Can be noisy early on
❌ Slightly slower

## Best Practices

1. **Use audio in demo mode** to hear final results
2. **Train silently** for speed and focus
3. **Monitor progress** visually during training
4. **Enjoy the music** once trained!

## Future Enhancements

Possible improvements:
- Real guitar samples instead of synthesis
- Note bending and vibrato
- Palm muting and harmonics
- Multiple instruments
- MIDI output support
- Recording/export to WAV

---

Enjoy listening to your AI learn guitar! 🎸🤖
