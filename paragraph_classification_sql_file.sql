-- database: ./classified_paragraphs_database.db

SELECT * from classified_paragraphs;


SELECT
    website_url AS URL,
    llm_model AS MODEL,
    AVG(computation_time) AS AVG_TIME -- No CAST needed here
FROM
    classified_paragraphs
GROUP BY
    website_url,
    llm_model
ORDER BY
    website_url,
    llm_model;


SELECT
    website_url AS URL,
    llm_model AS MODEL,
    prompt AS PROMPT_TEXT, -- Added prompt column with an alias
    AVG(computation_time) AS AVG_TIME
FROM
    classified_paragraphs
GROUP BY
    website_url,
    llm_model,
    prompt -- Added prompt to the grouping
ORDER BY
    website_url,
    llm_model,
    prompt; -- Added prompt to the ordering for consistent output


SELECT
    website_url AS URL,
    llm_model AS MODEL,
    prompt AS PROMPT_TEXT,
    score AS PARAGRAPH_SCORE, -- The specific score (1, 2, or 3)
    COUNT(*) AS SCORE_COUNT   -- Count of how many times this score occurred for this combination
FROM
    classified_paragraphs
GROUP BY
    website_url,
    llm_model,
    prompt,
    score -- Group by score to get counts for each score (1, 2, or 3)
ORDER BY
    website_url,
    llm_model,
    prompt,
    score;


SELECT
    primary_class.llm_model AS primary_model,
    other_class.llm_model AS comparing_model,
    primary_class.score AS classification_score,
    -- Count of paragraphs where both models agreed on this specific score
    COUNT(CASE WHEN primary_class.score = other_class.score THEN 1 END) AS agreed_count,
    -- Total count of paragraphs that the primary model classified with this score
    -- and that also exist in the comparing model's classifications
    COUNT(*) AS total_paragraphs_compared,
    -- Calculate the percentage of agreement
    CAST(COUNT(CASE WHEN primary_class.score = other_class.score THEN 1 END) AS REAL) * 100.0 / COUNT(*) AS agreement_percentage
FROM
    classified_paragraphs AS primary_class
JOIN
    classified_paragraphs AS other_class
    ON primary_class.website_url = other_class.website_url -- Ensure we compare paragraphs from the same URL
    AND primary_class.paragraph = other_class.paragraph   -- Ensure we compare the *same* paragraph content
WHERE
    primary_class.llm_model = 'mistral:latest' -- <<< Specify your PRIMARY model here
    AND other_class.llm_model != 'mistral:latest' -- Compare against all OTHER models
GROUP BY
    primary_class.llm_model,
    other_class.llm_model,
    primary_class.score
ORDER BY
    primary_class.llm_model,
    other_class.llm_model,
    primary_class.score;


SELECT
    primary_class.llm_model AS primary_model,
    other_class.llm_model AS comparing_model,
    primary_class.score AS classification_score,
    -- Count of paragraphs where both models agreed on this specific score
    COUNT(CASE WHEN primary_class.score = other_class.score THEN 1 END) AS agreed_count,
    -- Total count of paragraphs that the primary model classified with this score
    -- and that also exist in the comparing model's classifications
    COUNT(*) AS total_paragraphs_compared,
    -- Calculate the percentage of agreement
    CAST(COUNT(CASE WHEN primary_class.score = other_class.score THEN 1 END) AS REAL) * 100.0 / COUNT(*) AS agreement_percentage
FROM
    classified_paragraphs AS primary_class
JOIN
    classified_paragraphs AS other_class
    ON primary_class.website_url = other_class.website_url -- Ensure we compare paragraphs from the same URL
    AND primary_class.paragraph = other_class.paragraph   -- Ensure we compare the *same* paragraph content
WHERE
    primary_class.llm_model = 'mistral:latest' -- <<< Specify your PRIMARY model here
    AND other_class.llm_model != 'mistral:latest' -- Compare against all OTHER models
GROUP BY
    primary_class.llm_model,
    other_class.llm_model,
    primary_class.score
ORDER BY
    primary_class.llm_model,
    other_class.llm_model,
    primary_class.score;


SELECT
    primary_class.llm_model AS primary_model,
    other_class.llm_model AS comparing_model,
    primary_class.prompt AS prompt_structure, -- Added prompt structure to select
    primary_class.score AS classification_score,
    COUNT(CASE WHEN primary_class.score = other_class.score THEN 1 END) AS agreed_count,
    COUNT(*) AS total_paragraphs_compared,
    CAST(COUNT(CASE WHEN primary_class.score = other_class.score THEN 1 END) AS REAL) * 100.0 / COUNT(*) AS agreement_percentage
FROM
    classified_paragraphs AS primary_class
JOIN
    classified_paragraphs AS other_class
    ON primary_class.website_url = other_class.website_url
    AND primary_class.paragraph = other_class.paragraph
    AND primary_class.prompt = other_class.prompt -- <<< KEY ADDITION: Join on prompt structure
WHERE
    primary_class.llm_model = 'mistral:latest' -- <<< Specify your PRIMARY model here
    AND other_class.llm_model != 'mistral:latest' -- Compare against all OTHER models
GROUP BY
    primary_class.llm_model,
    other_class.llm_model,
    primary_class.prompt, -- Added prompt structure to grouping
    primary_class.score
ORDER BY
    primary_class.llm_model,
    other_class.llm_model,
    primary_class.prompt,
    primary_class.score;