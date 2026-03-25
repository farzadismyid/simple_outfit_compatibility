# SOC, Simple Outfit Compatibility

A simple starter project for outfit compatibility and complementary item recommendation.

## Goal
Given an outfit image or a set of outfit item images, recommend the top 5 compatible items from a local catalog.

## Version 1
- Use CLIP embeddings
- Build a small item catalog
- Retrieve top 5 compatible items by cosine similarity
- Filter by target category such as shoes, bag, jacket

## Project Structure
- `notebooks/` experiments and step-by-step workflow
- `src/soc/` source code
- `data/` local images and metadata
- `outputs/` saved embeddings, predictions, figures

## First Milestone
1. Load catalog images
2. Create CLIP embeddings
3. Save embeddings
4. Input a query outfit
5. Recommend top 5 items