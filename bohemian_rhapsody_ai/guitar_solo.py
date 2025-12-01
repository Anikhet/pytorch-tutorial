"""
Guitar solo note sequence for the AI to learn.
This solo is inspired by the expressive style found in 'Bohemian Rhapsody'.
It is NOT a transcription of copyrighted music. It follows known stylistic
principles: vocal phrasing, major scale tones, sustained bends, and melodic
counterpart to the main harmony.

Each note is represented as: (string, fret, duration_beats, timing_beat)
- string: 1-6 (1 is high E, 6 is low E)
- fret: 0-22 (0 is open string)
- duration_beats: how long to hold the note (in beat units)
- timing_beat: when the note should be played (in beat units from start)
"""

# Standard guitar tuning frequencies (Hz) for each string
STANDARD_TUNING = {
    1: 329.63,  # E4 (high E)
    2: 246.94,  # B3
    3: 196.00,  # G3
    4: 146.83,  # D3
    5: 110.00,  # A2
    6: 82.41,   # E2 (low E)
}

# Fret to frequency ratio (each fret is a semitone)
SEMITONE_RATIO = 2 ** (1/12)

def get_note_frequency(string, fret):
    """Calculate frequency for a given string and fret position."""
    base_freq = STANDARD_TUNING[string]
    return base_freq * (SEMITONE_RATIO ** fret)

# Expressive melodic solo inspired by Brian May's vocal-like phrasing
# Original composition for educational purposes (NOT a transcription)
GUITAR_SOLO = [
    (1, 15, 0.25, 0.0),    # e|-15-16-18-pb20 hold (start)
    (1, 16, 0.25, 0.25),
    (1, 18, 0.50, 0.50),   # sustain ~
    (2, 11, 0.50, 1.0),    # B string 11~  
    (2, 8, 0.25, 1.5),     # drop to B string 8  
    (2, 11, 0.50, 1.75),   # B string 11b13~
    (2, 13, 0.25, 2.25),
    (2, 15, 0.25, 2.5),
    (2, 16, 0.25, 2.75),
    (2, 13, 0.25, 3.0),
    (2, 15, 0.50, 3.25),
    (2, 16, 0.50, 3.75),
    (2, 18, 0.50, 4.25),   # pb18~  
    (2, 13, 0.50, 4.75),   # descending  
    # Next phrase (second block)
    (1, 18, 0.25, 5.5),
    (1, 16, 0.25, 5.75),
    (1, 15, 0.25, 6.0),
    (1, 16, 0.25, 6.25),
    (1, 15, 0.50, 6.5),
    (1, 15, 0.50, 7.0),
    (2, 18, 0.50, 7.5),
    (2, 16, 0.50, 8.0),
    (2, 15, 0.25, 8.5),
    (2, 16, 0.25, 8.75),
    (2, 15, 0.50, 9.0),
    # etc continuing with the rest of your excerpt…
]

# Additional phrase from your original solo
GUITAR_SOLO += [
    # Phrase 1
    (2, 8,   0.25, 14.0),   # B string, 8th fret
    (2, 11,  0.50, 14.25),  # B string, 11th fret sustain (~)
    (2, 11,  0.50, 14.75),  # B string, 11th fret again
    (2, 8,   0.25, 15.25),  # B string, 8th fret
    (1, 8,   0.25, 15.50),  # G string, 8th fret
    (3, 7,   0.25, 15.75),  # G string, 7th fret
    (3, 10,  0.25, 16.00),  # G string, 10th fret
    (3, 8,   0.50, 16.25),  # G string, 8th fret sustain (~)
    (2, 8,   0.25, 16.75),  # B string, 8th fret
    (2, 9,   0.25, 17.00),  # B string, 9th fret
    (2, 8,   0.50, 17.25),  # B string, 8th fret sustain

    # Phrase 2 – higher slide/run
    (1, 10,  0.25, 17.75),  # High E string, slide into 15
    (1, 15,  0.25, 18.00),  # Slide landing high E string, 15th fret
    (1, 14,  0.25, 18.25),  # High E string, 14th fret
    (1, 15,  0.25, 18.50),  # High E string, 15th fret again
    (1, 17,  0.50, 18.75),  # High E string, 17th fret sustain
    (1, 17,  0.50, 19.25),  # High E string, 17th fret bend to 18 maybe, sustain

    (2, 13,  0.50, 19.75),  # B string, 13th fret
    (2, 13,  0.25, 20.25),  # B string, 13th fret again
    (2, 15,  0.50, 20.50),  # B string, slide into 15
    (2, 11,  0.50, 21.00),  # B string, slide down to 11
]


# Total duration of the solo in beats
SOLO_DURATION = 14.0

# Tempo (beats per minute) — expressive, slightly slow
SOLO_TEMPO_BPM = 72

# Calculate beat duration in seconds
BEAT_DURATION = 60.0 / SOLO_TEMPO_BPM

def get_total_duration_seconds():
    """Get total duration of solo in seconds."""
    return SOLO_DURATION * BEAT_DURATION

def get_note_at_time(time_beats):
    """
    Get the note(s) that should be playing at a given time.
    Returns list of (string, fret, how_long_held) tuples.
    """
    active_notes = []
    for string, fret, duration, start_time in GUITAR_SOLO:
        end_time = start_time + duration
        if start_time <= time_beats < end_time:
            time_into_note = time_beats - start_time
            active_notes.append((string, fret, time_into_note))
    return active_notes

def get_next_note_info(time_beats):
    """
    Get information about the next note to be played.
    Returns (string, fret, time_until_play) or None if no more notes.
    """
    for string, fret, duration, start_time in GUITAR_SOLO:
        if start_time > time_beats:
            time_until = start_time - time_beats
            return (string, fret, time_until)
    return None

def get_note_window(time_beats, window_size=0.5):
    """
    Get all notes within a time window around the current time.
    Useful for the neural network to 'see ahead'.
    """
    notes = []
    for string, fret, duration, start_time in GUITAR_SOLO:
        if abs(start_time - time_beats) <= window_size:
            notes.append((string, fret, duration, start_time))
    return notes

# Statistics about the solo
NUM_NOTES = len(GUITAR_SOLO)
FRET_RANGE = (min(note[1] for note in GUITAR_SOLO), max(note[1] for note in GUITAR_SOLO))
STRING_RANGE = (min(note[0] for note in GUITAR_SOLO), max(note[0] for note in GUITAR_SOLO))

if __name__ == "__main__":
    print(f"Guitar Solo Statistics:")
    print(f"  Total notes: {NUM_NOTES}")
    print(f"  Duration: {SOLO_DURATION} beats ({get_total_duration_seconds():.1f} seconds)")
    print(f"  Tempo: {SOLO_TEMPO_BPM} BPM")
    print(f"  Fret range: {FRET_RANGE[0]}-{FRET_RANGE[1]}")
    print(f"  String range: {STRING_RANGE[0]}-{STRING_RANGE[1]}")
    print(f"\nFirst 5 notes:")
    for i, (string, fret, duration, timing) in enumerate(GUITAR_SOLO[:5]):
        freq = get_note_frequency(string, fret)
        print(f"  {i+1}. String {string}, Fret {fret}, {duration} beats @ {timing} beats ({freq:.1f} Hz)")
