package preload

// Words returns a curated list designed to form distinct semantic clusters.
// Categories: animals, colors, emotions, food, music, sports, weather, tech.
func Words() []string {
	return []string{
		// Animals
		"dog", "cat", "wolf", "lion", "tiger", "elephant", "giraffe", "zebra",
		"eagle", "hawk", "sparrow", "penguin", "dolphin", "whale", "shark", "salmon",

		// Colors
		"red", "blue", "green", "yellow", "purple", "orange", "pink", "black",
		"white", "gray", "crimson", "azure", "emerald", "gold", "silver", "indigo",

		// Emotions
		"happy", "sad", "angry", "fearful", "surprised", "disgusted", "anxious", "calm",
		"excited", "bored", "grateful", "jealous", "proud", "ashamed", "hopeful", "melancholy",

		// Food
		"pizza", "burger", "sushi", "pasta", "salad", "steak", "bread", "cheese",
		"apple", "banana", "orange", "grape", "strawberry", "chocolate", "cake", "ice cream",

		// Music
		"guitar", "piano", "drums", "violin", "trumpet", "flute", "bass", "saxophone",
		"jazz", "rock", "classical", "blues", "hip hop", "country", "metal", "electronic",

		// Sports
		"soccer", "basketball", "tennis", "golf", "baseball", "hockey", "football", "volleyball",
		"swimming", "running", "cycling", "boxing", "wrestling", "skiing", "surfing", "climbing",

		// Weather
		"sunny", "rainy", "cloudy", "snowy", "windy", "foggy", "stormy", "humid",
		"freezing", "scorching", "drizzle", "thunder", "lightning", "hail", "frost", "drought",

		// Tech
		"computer", "keyboard", "monitor", "mouse", "server", "database", "algorithm", "network",
		"internet", "software", "hardware", "compiler", "debugger", "terminal", "browser", "encryption",
	}
}
