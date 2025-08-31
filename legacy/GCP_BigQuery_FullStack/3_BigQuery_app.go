package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"

	"cloud.google.com/go/bigquery"
	"github.com/joho/godotenv"
	"google.golang.org/api/iterator"
)

func main() {
	// Load environment variables
	err := godotenv.Load()
	if err != nil {
		log.Fatal("Error loading .env file")
	}

	// Set up HTTP handler
	http.HandleFunc("/", handleColumns)

	fmt.Println("Server starting on :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}

func handleColumns(w http.ResponseWriter, r *http.Request) {
	ctx := context.Background()

	// Get project ID from environment
	projectID := os.Getenv("GCP_PROJECT_ID")
	if projectID == "" {
		http.Error(w, "GCP_PROJECT_ID environment variable not set", http.StatusInternalServerError)
		return
	}

	// Create BigQuery client
	client, err := bigquery.NewClient(ctx, projectID)
	if err != nil {
		http.Error(w, "Failed to create BigQuery client: "+err.Error(), http.StatusInternalServerError)
		return
	}
	defer client.Close()

	// Set content type
	w.Header().Set("Content-Type", "text/html")

	// Write HTML response
	fmt.Fprintf(w, `
<!DOCTYPE html>
<html>
<head>
    <title>BigQuery Column Names</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1 { color: #333; }
        ul { list-style-type: none; padding: 0; }
        li { background: #f4f4f4; margin: 5px 0; padding: 10px; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>Product Pairs Table - Column Names</h1>
`)

	// List all datasets first
	fmt.Fprintf(w, "<h2>Available datasets:</h2><ul>")
	datasets := client.Datasets(ctx)
	for {
		dataset, err := datasets.Next()
		if err == iterator.Done {
			break
		}
		if err != nil {
			fmt.Fprintf(w, "<li>Error listing datasets: %v</li>", err)
			break
		}
		fmt.Fprintf(w, "<li>%s</li>", dataset.DatasetID)
	}
	fmt.Fprintf(w, "</ul>")

	// Get table schema for product_pairs
	dataset := client.Dataset("my_dataset")
	table := dataset.Table("product_pairs")

	metadata, err := table.Metadata(ctx)
	if err != nil {
		fmt.Fprintf(w, "<p>Error getting table metadata: %v</p>", err)
	} else {
		fmt.Fprintf(w, "<h2>Product Pairs Columns:</h2><ul>")
		for _, field := range metadata.Schema {
			fmt.Fprintf(w, "<li>%s (%s)</li>", field.Name, field.Type)
		}
		fmt.Fprintf(w, "</ul>")
		fmt.Fprintf(w, "<p><strong>Total rows:</strong> %d</p>", metadata.NumRows)
	}

	fmt.Fprintf(w, `
    <p><em>Connected to BigQuery successfully!</em></p>
</body>
</html>
`)
}
