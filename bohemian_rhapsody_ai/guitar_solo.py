"""
Guitar solo note sequence for the AI to learn.
This is a rock-style guitar solo sequence inspired by classic rock patterns.

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

# Rock-style guitar solo sequence (inspired by classic rock patterns)
# This is an original composition for educational purposes
GUITAR_SOLO = [
    # Opening phrase - ascending scale run
    (3, 5, 0.25, 0.0),    # G string, 5th fret (C)
    (3, 7, 0.25, 0.25),   # G string, 7th fret (D)
    (3, 9, 0.25, 0.5),    # G string, 9th fret (E)
    (2, 10, 0.25, 0.75),  # B string, 10th fret (A)
    (2, 12, 0.5, 1.0),    # B string, 12th fret (B) - hold

    # Fast descending run
    (2, 10, 0.25, 1.5),   # B string, 10th fret
    (3, 9, 0.25, 1.75),   # G string, 9th fret
    (3, 7, 0.25, 2.0),    # G string, 7th fret
    (3, 5, 0.25, 2.25),   # G string, 5th fret

    # Bends and sustained notes
    (2, 8, 1.0, 2.5),     # B string, 8th fret (G) - sustained bend
    (1, 12, 0.5, 3.5),    # High E, 12th fret (E) - high note
    (1, 15, 0.5, 4.0),    # High E, 15th fret (G)

    # Triplet phrase
    (2, 13, 0.17, 4.5),   # Fast triplet
    (2, 12, 0.17, 4.67),
    (2, 10, 0.17, 4.84),

    # Power chord hits (same time, different strings)
    (3, 5, 0.5, 5.0),     # G string
    (4, 5, 0.5, 5.0),     # D string (same fret = power chord)

    # Melodic phrase
    (2, 8, 0.5, 5.5),     # B string, 8th fret
    (2, 10, 0.5, 6.0),    # B string, 10th fret
    (3, 9, 0.5, 6.5),     # G string, 9th fret
    (3, 7, 0.5, 7.0),     # G string, 7th fret

    # Fast run to climax
    (1, 12, 0.25, 7.5),   # High E
    (1, 13, 0.25, 7.75),
    (1, 15, 0.25, 8.0),
    (1, 17, 0.25, 8.25),

    # Climax - high sustained note
    (1, 19, 1.5, 8.5),    # High E, 19th fret - big hold

    # Descending ending phrase
    (2, 15, 0.5, 10.0),   # B string
    (2, 13, 0.5, 10.5),
    (2, 12, 0.5, 11.0),
    (2, 10, 0.5, 11.5),

    # Final resolution
    (3, 9, 0.5, 12.0),    # G string, 9th fret
    (3, 7, 0.5, 12.5),
    (3, 5, 1.0, 13.0),    # Final note - hold
]

# Total duration of the solo in beats
SOLO_DURATION = 14.0

# Tempo (beats per minute)
SOLO_TEMPO_BPM = 120

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
