package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"text/tabwriter"
)

func main() {
	const path = "/home/ubuntu/projects/Portfolio/horizon3_funding_lead_structured.csv"

	// 1. Open the file
	file, err := os.Open(path)
	if err != nil {
		log.Fatalf("could not open %s: %v", path, err)
	}
	defer file.Close()

	// 2. Read all rows
	r := csv.NewReader(file)
	records, err := r.ReadAll()
	if err != nil {
		log.Fatalf("csv read error: %v", err)
	}

	// 3. Print as a table
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	for i, row := range records {
		for j, col := range row {
			if j > 0 {
				fmt.Fprint(w, "\t") // tab between columns
			}
			fmt.Fprint(w, col)
		}
		fmt.Fprintln(w) // newline at end of row

		// Optional: draw a separator line after the header
		if i == 0 {
			for k := range row {
				if k > 0 {
					fmt.Fprint(w, "\t")
				}
				fmt.Fprint(w, "----------")
			}
			fmt.Fprintln(w)
		}
	}
	w.Flush()
}
