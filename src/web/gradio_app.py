# gradio_app.py
"""
Gradio Frontend for Soma ML Platform
Interactive dashboard for demand forecasting and book recommendations
"""

import json
import os
import sys
from datetime import datetime, timedelta

import duckdb
import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests

# Add ml_models to path
sys.path.insert(0, "./ml_models")


class SomaGradioApp:
    """Main Gradio application for Soma"""

    def __init__(self):
        self.db_path = "./data/soma.duckdb"
        self.api_base_url = "http://localhost:5001"
        self.conn = None
        self.setup_database()

    def setup_database(self):
        """Setup database connection"""
        try:
            if os.path.exists(self.db_path):
                self.conn = duckdb.connect(self.db_path)
                print("✅ Connected to DuckDB")
            else:
                print(
                    "❌ Database not found. Please run generate_synthetic_data.py first"
                )
        except Exception as e:
            print(f"❌ Database connection error: {e}")

    def get_book_list(self):
        """Get list of books for dropdown"""
        try:
            if not self.conn:
                return ["Database not available"]

            books = self.conn.execute(
                """
                SELECT book_id, title, genre, price 
                FROM dim_books 
                ORDER BY title 
                LIMIT 100
            """
            ).df()

            if len(books) > 0:
                return [
                    f"{row['book_id']} - {row['title']}" for _, row in books.iterrows()
                ]
            else:
                return ["No books available"]
        except Exception as e:
            print(f"Error getting book list: {e}")
            return ["Error loading books"]

    def get_book_id_from_selection(self, selection):
        """Extract book ID from dropdown selection"""
        if " - " in selection:
            return selection.split(" - ")[0]
        return selection

    def predict_demand(self, book_selection, days_ahead, model_params):
        """Predict demand for selected book"""
        try:
            book_id = self.get_book_id_from_selection(book_selection)

            # Try API first
            try:
                response = requests.post(
                    f"{self.api_base_url}/predict/demand",
                    json={"book_id": book_id, "days_ahead": days_ahead},
                    timeout=10,
                )

                if response.status_code == 200:
                    data = response.json()

                    # Create forecast visualization
                    fig = self.create_demand_forecast_plot(data)

                    # Create summary table
                    summary_df = pd.DataFrame(data["predictions"])

                    result_text = f"""
                    ## Demand Forecast Results
                    
                    **Book:** {data.get('book_title', book_id)}
                    **Forecast Period:** {days_ahead} days
                    **Model Confidence:** {data.get('model_confidence', 'Unknown')}
                    **Generated:** {data.get('generated_at', 'Unknown')}
                    
                    ### Summary Statistics
                    - **Average Daily Demand:** {summary_df['predicted_demand'].mean():.1f} units
                    - **Total Predicted Demand:** {summary_df['predicted_demand'].sum():.0f} units
                    - **Peak Day:** Day {summary_df.loc[summary_df['predicted_demand'].idxmax(), 'day']}
                    """

                    return result_text, fig, summary_df

            except requests.exceptions.RequestException:
                pass  # Fall back to local prediction

            # Fallback to local prediction
            return self.fallback_demand_prediction(book_id, days_ahead)

        except Exception as e:
            error_msg = f"Error predicting demand: {str(e)}"
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                text="Error generating forecast",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
            )
            empty_df = pd.DataFrame()
            return error_msg, empty_fig, empty_df

    def fallback_demand_prediction(self, book_id, days_ahead):
        """Fallback demand prediction using database data"""
        try:
            # Get book info
            book_info = self.conn.execute(
                """
                SELECT title, genre, price FROM dim_books WHERE book_id = ?
            """,
                [book_id],
            ).df()

            if len(book_info) == 0:
                return "Book not found", go.Figure(), pd.DataFrame()

            # Get historical sales for similar books
            avg_demand = (
                self.conn.execute(
                    """
                SELECT AVG(quantity) as avg_qty
                FROM fact_sales s
                JOIN dim_books b ON s.book_id = b.book_id
                WHERE b.genre = ?
            """,
                    [book_info.iloc[0]["genre"]],
                ).fetchone()[0]
                or 5
            )

            # Generate simple forecast
            predictions = []
            for day in range(1, days_ahead + 1):
                seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * day / 7)
                noise = np.random.uniform(0.8, 1.2)
                demand = max(1, int(avg_demand * seasonal_factor * noise))

                predictions.append(
                    {
                        "day": day,
                        "predicted_demand": demand,
                        "confidence_lower": max(1, int(demand * 0.7)),
                        "confidence_upper": int(demand * 1.3),
                    }
                )

            summary_df = pd.DataFrame(predictions)

            # Create plot
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=summary_df["day"],
                    y=summary_df["predicted_demand"],
                    mode="lines+markers",
                    name="Predicted Demand",
                    line=dict(color="blue", width=2),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=summary_df["day"],
                    y=summary_df["confidence_upper"],
                    fill=None,
                    mode="lines",
                    line_color="rgba(0,100,80,0)",
                    showlegend=False,
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=summary_df["day"],
                    y=summary_df["confidence_lower"],
                    fill="tonexty",
                    mode="lines",
                    line_color="rgba(0,100,80,0)",
                    name="Confidence Interval",
                    fillcolor="rgba(0,100,80,0.2)",
                )
            )

            fig.update_layout(
                title=f"Demand Forecast for {book_info.iloc[0]['title']}",
                xaxis_title="Days Ahead",
                yaxis_title="Predicted Demand (units)",
                hovermode="x unified",
            )

            result_text = f"""
            ## Demand Forecast Results (Fallback Model)
            
            **Book:** {book_info.iloc[0]['title']}
            **Genre:** {book_info.iloc[0]['genre']}
            **Price:** ${book_info.iloc[0]['price']:.2f}
            **Forecast Period:** {days_ahead} days
            
            ### Summary Statistics
            - **Average Daily Demand:** {summary_df['predicted_demand'].mean():.1f} units
            - **Total Predicted Demand:** {summary_df['predicted_demand'].sum():.0f} units
            - **Based on genre average:** {avg_demand:.1f} units/day
            """

            return result_text, fig, summary_df

        except Exception as e:
            error_msg = f"Fallback prediction error: {str(e)}"
            return error_msg, go.Figure(), pd.DataFrame()

    def create_demand_forecast_plot(self, forecast_data):
        """Create plotly visualization for demand forecast"""
        df = pd.DataFrame(forecast_data["predictions"])

        fig = go.Figure()

        # Add main forecast line
        fig.add_trace(
            go.Scatter(
                x=df["day"],
                y=df["predicted_demand"],
                mode="lines+markers",
                name="Predicted Demand",
                line=dict(color="#1f77b4", width=3),
                marker=dict(size=8),
            )
        )

        # Add confidence intervals if available
        if "confidence_interval" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["day"],
                    y=[ci["upper"] for ci in df["confidence_interval"]],
                    fill=None,
                    mode="lines",
                    line_color="rgba(0,100,80,0)",
                    showlegend=False,
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=df["day"],
                    y=[ci["lower"] for ci in df["confidence_interval"]],
                    fill="tonexty",
                    mode="lines",
                    line_color="rgba(0,100,80,0)",
                    name="Confidence Interval",
                    fillcolor="rgba(31,119,180,0.2)",
                )
            )

        fig.update_layout(
            title=f"Demand Forecast: {forecast_data.get('book_title', 'Unknown Book')}",
            xaxis_title="Days Ahead",
            yaxis_title="Predicted Demand (units)",
            hovermode="x unified",
            height=400,
            showlegend=True,
        )

        return fig

    def get_recommendations(self, book_selection, user_type, n_recommendations):
        """Get book recommendations"""
        try:
            book_id = (
                self.get_book_id_from_selection(book_selection)
                if book_selection
                else None
            )

            # Try API first
            try:
                payload = {"n_recommendations": n_recommendations}
                if book_id:
                    payload["book_id"] = book_id
                if user_type and user_type != "Any":
                    payload["user_type"] = user_type

                response = requests.post(
                    f"{self.api_base_url}/recommend", json=payload, timeout=10
                )

                if response.status_code == 200:
                    data = response.json()
                    return self.format_recommendations(data, book_id)

            except requests.exceptions.RequestException:
                pass  # Fall back to local recommendations

            # Fallback to database recommendations
            return self.fallback_recommendations(book_id, user_type, n_recommendations)

        except Exception as e:
            return f"Error getting recommendations: {str(e)}", pd.DataFrame()

    def fallback_recommendations(self, book_id, user_type, n_recommendations):
        """Fallback recommendations using database queries"""
        try:
            if book_id:
                # Content-based recommendations by genre
                book_genre = self.conn.execute(
                    """
                    SELECT genre FROM dim_books WHERE book_id = ?
                """,
                    [book_id],
                ).fetchone()

                if book_genre:
                    similar_books = self.conn.execute(
                        """
                        SELECT 
                            b.book_id,
                            b.title,
                            b.genre,
                            b.price,
                            b.price_category,
                            COALESCE(SUM(s.quantity), 0) as popularity
                        FROM dim_books b
                        LEFT JOIN fact_sales s ON b.book_id = s.book_id
                        WHERE b.genre = ? AND b.book_id != ?
                        GROUP BY b.book_id, b.title, b.genre, b.price, b.price_category
                        ORDER BY popularity DESC, b.title
                        LIMIT ?
                    """,
                        [book_genre[0], book_id, n_recommendations],
                    ).df()
                else:
                    similar_books = pd.DataFrame()
            else:
                # Popular books overall
                similar_books = self.conn.execute(
                    """
                    SELECT 
                        b.book_id,
                        b.title,
                        b.genre,
                        b.price,
                        b.price_category,
                        COALESCE(SUM(s.quantity), 0) as popularity
                    FROM dim_books b
                    LEFT JOIN fact_sales s ON b.book_id = s.book_id
                    GROUP BY b.book_id, b.title, b.genre, b.price, b.price_category
                    ORDER BY popularity DESC, b.title
                    LIMIT ?
                """,
                    [n_recommendations],
                ).df()

            if len(similar_books) == 0:
                return "No recommendations found", pd.DataFrame()

            # Format recommendations
            recommendations_text = f"""
            ## Book Recommendations
            
            **Based on:** {"Similar genre" if book_id else "Overall popularity"}
            **Method:** Database query (fallback)
            **Found:** {len(similar_books)} recommendations
            """

            # Prepare display dataframe
            display_df = similar_books[
                ["title", "genre", "price", "price_category", "popularity"]
            ].copy()
            display_df.columns = [
                "Title",
                "Genre",
                "Price ($)",
                "Price Category",
                "Popularity Score",
            ]
            display_df["Price ($)"] = display_df["Price ($)"].round(2)

            return recommendations_text, display_df

        except Exception as e:
            return f"Fallback recommendations error: {str(e)}", pd.DataFrame()

    def format_recommendations(self, api_data, book_id):
        """Format API recommendation response"""
        recommendations = api_data.get("recommendations", [])

        if not recommendations:
            return "No recommendations found", pd.DataFrame()

        recommendations_text = f"""
        ## Book Recommendations
        
        **Method:** {api_data.get('method', 'API')}
        **Generated:** {api_data.get('generated_at', 'Unknown')}
        **Total Found:** {api_data.get('total_found', len(recommendations))}
        """

        # Create DataFrame for display
        df_data = []
        for rec in recommendations:
            df_data.append(
                {
                    "Title": rec.get("title", "Unknown"),
                    "Genre": rec.get("genre", "Unknown"),
                    "Confidence Score": rec.get(
                        "confidence_score", rec.get("similarity_score", "N/A")
                    ),
                    "Reason": rec.get("reason", "Recommended"),
                }
            )

        display_df = pd.DataFrame(df_data)

        return recommendations_text, display_df

    def get_book_analytics(self, book_selection):
        """Get comprehensive book analytics"""
        try:
            book_id = self.get_book_id_from_selection(book_selection)

            # Get book details
            book_info = self.conn.execute(
                """
                SELECT * FROM dim_books WHERE book_id = ?
            """,
                [book_id],
            ).df()

            if len(book_info) == 0:
                return "Book not found", go.Figure(), pd.DataFrame()

            book = book_info.iloc[0]

            # Get sales analytics
            sales_data = self.conn.execute(
                """
                SELECT 
                    DATE_TRUNC('month', sale_date) as month,
                    SUM(quantity) as total_quantity,
                    SUM(total_amount) as total_revenue,
                    COUNT(*) as transactions,
                    AVG(unit_price) as avg_price
                FROM fact_sales
                WHERE book_id = ?
                GROUP BY DATE_TRUNC('month', sale_date)
                ORDER BY month
            """,
                [book_id],
            ).df()

            # Create analytics visualization
            if len(sales_data) > 0:
                fig = go.Figure()

                # Add sales quantity line
                fig.add_trace(
                    go.Scatter(
                        x=sales_data["month"],
                        y=sales_data["total_quantity"],
                        mode="lines+markers",
                        name="Units Sold",
                        yaxis="y",
                        line=dict(color="blue"),
                    )
                )

                # Add revenue line on secondary y-axis
                fig.add_trace(
                    go.Scatter(
                        x=sales_data["month"],
                        y=sales_data["total_revenue"],
                        mode="lines+markers",
                        name="Revenue ($)",
                        yaxis="y2",
                        line=dict(color="red"),
                    )
                )

                fig.update_layout(
                    title=f"Sales Performance: {book['title']}",
                    xaxis_title="Month",
                    yaxis=dict(title="Units Sold", side="left"),
                    yaxis2=dict(title="Revenue ($)", side="right", overlaying="y"),
                    hovermode="x unified",
                    height=400,
                )
            else:
                fig = go.Figure()
                fig.add_annotation(
                    text="No sales data available",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                )

            # Create summary analytics
            analytics_text = f"""
            ## Book Analytics: {book['title']}
            
            ### Basic Information
            - **Author:** {book.get('author', 'Unknown')}
            - **Genre:** {book['genre']}
            - **Price:** ${book['price']:.2f} ({book.get('price_category', 'Unknown')} category)
            - **Format:** {book.get('format', 'Unknown')}
            - **Pages:** {book.get('page_count', 'Unknown')} ({book.get('length_category', 'Unknown')} length)
            - **Publication Year:** {book.get('publication_year', 'Unknown')}
            
            ### Sales Performance
            """

            if len(sales_data) > 0:
                total_units = sales_data["total_quantity"].sum()
                total_revenue = sales_data["total_revenue"].sum()
                avg_monthly_sales = sales_data["total_quantity"].mean()

                analytics_text += f"""
                - **Total Units Sold:** {total_units:,.0f}
                - **Total Revenue:** ${total_revenue:,.2f}
                - **Average Monthly Sales:** {avg_monthly_sales:.1f} units
                - **Average Transaction Value:** ${total_revenue/total_units if total_units > 0 else 0:.2f}
                - **Months with Sales:** {len(sales_data)}
                """

                # Create summary table
                summary_df = sales_data.copy()
                summary_df["month"] = summary_df["month"].dt.strftime("%Y-%m")
                summary_df.columns = [
                    "Month",
                    "Units Sold",
                    "Revenue ($)",
                    "Transactions",
                    "Avg Price ($)",
                ]
                summary_df = summary_df.round(2)
            else:
                analytics_text += "\n- **No sales data available**"
                summary_df = pd.DataFrame()

            return analytics_text, fig, summary_df

        except Exception as e:
            error_msg = f"Error getting book analytics: {str(e)}"
            return error_msg, go.Figure(), pd.DataFrame()

    def generate_ad_copy_ui(
        self, book_selection, ad_type, target_audience, style_preference
    ):
        """Generate ad copy for selected book"""
        try:
            book_id = self.get_book_id_from_selection(book_selection)

            # Try API first
            try:
                response = requests.post(
                    f"{self.api_base_url}/generate/ad-copy",
                    json={
                        "book_id": book_id,
                        "ad_type": ad_type,
                        "target_audience": (
                            target_audience
                            if target_audience != "Auto-detect"
                            else None
                        ),
                    },
                    timeout=10,
                )

                if response.status_code == 200:
                    data = response.json()
                    return self.format_ad_copy_results(data)

            except requests.exceptions.RequestException as e:
                print(f"API request failed: {e}")

            # Fallback to local generation
            return self.fallback_ad_copy_generation(book_id, ad_type, target_audience)

        except Exception as e:
            error_msg = f"Error generating ad copy: {str(e)}"
            empty_df = pd.DataFrame()
            return error_msg, empty_df

    def format_ad_copy_results(self, api_data):
        """Format API ad copy response for display"""
        if "error" in api_data:
            return f"Error: {api_data['error']}", pd.DataFrame()

        # Create main results text
        results_text = f"""
        ## 🎯 Ad Copy Generated Successfully
        
        **Book:** {api_data.get('book_id', 'Unknown')}
        **Ad Type:** {api_data.get('ad_type', 'social_media').title()}
        **Target Audience:** {api_data.get('target_audience', 'General')}
        **Generated:** {api_data.get('generated_at', 'Unknown')}
        **Similar Books Context:** {', '.join(api_data.get('similar_books_context', []))}
        
        ### 📊 Summary
        - **Variants Generated:** {len(api_data.get('ad_copy_variants', []))}
        - **Average Length:** {np.mean([v.get('length', 0) for v in api_data.get('ad_copy_variants', [])]) if api_data.get('ad_copy_variants') else 0:.0f} characters
        """

        # Create DataFrame for variants
        variants_data = []
        for i, variant in enumerate(api_data.get("ad_copy_variants", [])):
            variants_data.append(
                {
                    "Variant": f"#{i+1}",
                    "Ad Copy": variant.get("text", ""),
                    "Type": variant.get("type", "template").title(),
                    "Confidence": f"{variant.get('confidence', 0.0):.1%}",
                    "Length": variant.get("length", 0),
                    "Call to Action": variant.get("call_to_action", ""),
                }
            )

        variants_df = pd.DataFrame(variants_data)

        return results_text, variants_df

    def fallback_ad_copy_generation(self, book_id, ad_type, target_audience):
        """Fallback ad copy generation using database"""
        try:
            # Get book info
            book_info = self.conn.execute(
                """
                SELECT title, author, genre, price FROM dim_books WHERE book_id = ?
            """,
                [book_id],
            ).df()

            if len(book_info) == 0:
                return "Book not found", pd.DataFrame()

            book = book_info.iloc[0]

            # Simple template-based generation
            templates = {
                "social_media": [
                    f"📚 Discover {book['title']} by {book['author']} - A captivating {book['genre'].lower()} story! ✨ #BookLovers #ReadingCommunity",
                    f"🔥 New read alert! {book['title']} will take you on an unforgettable journey. Get your copy today! 📖",
                    f"⭐ Why readers love {book['title']}: Engaging plot, brilliant writing, {book['genre']} at its best! 🎯",
                ],
                "email": [
                    f"Subject: Don't Miss Out - {book['title']} Now Available\n\nDear Reader, Experience the compelling world of {book['title']} by {book['author']}. This {book['genre'].lower()} masterpiece is priced at just ${book['price']:.2f}.",
                    f"Subject: Your Next Great Read Awaits\n\n{book['title']} by {book['author']} has arrived! Join thousands of readers who've already discovered this {book['genre'].lower()} gem.",
                ],
                "display": [
                    f"{book['title']} - The {book['genre']} Novel Everyone's Talking About",
                    f"From acclaimed author {book['author']} comes {book['title']} - Available Now",
                ],
            }

            selected_templates = templates.get(ad_type, templates["social_media"])

            # Create variants
            variants_data = []
            for i, template in enumerate(selected_templates):
                variants_data.append(
                    {
                        "Variant": f"#{i+1}",
                        "Ad Copy": template,
                        "Type": "Template",
                        "Confidence": "80%",
                        "Length": len(template),
                        "Call to Action": (
                            f"Order ${book['genre']} now!"
                            if ad_type == "social_media"
                            else "Learn More"
                        ),
                    }
                )

            results_text = f"""
            ## 🎯 Ad Copy Generated (Fallback Mode)
            
            **Book:** {book['title']} by {book['author']}
            **Genre:** {book['genre']}
            **Price:** ${book['price']:.2f}
            **Ad Type:** {ad_type.title()}
            **Target Audience:** {target_audience or 'General'}
            
            ### 📊 Summary
            - **Variants Generated:** {len(variants_data)}
            - **Method:** Template-based (fallback)
            """

            variants_df = pd.DataFrame(variants_data)
            return results_text, variants_df

        except Exception as e:
            return f"Fallback generation error: {str(e)}", pd.DataFrame()

    def fallback_image_prompts_generation(self, book_id, style, dimensions):
        """Fallback image prompts generation using database"""
        try:
            # Get book info
            book_info = self.conn.execute(
                """
                SELECT title, author, genre, price FROM dim_books WHERE book_id = ?
            """,
                [book_id],
            ).df()

            if len(book_info) == 0:
                return "Book not found", pd.DataFrame()

            book = book_info.iloc[0]

            # Simple template-based generation
            prompts = {
                "modern": [
                    f"Modern book cover design for '{book['title']}' by {book['author']} - clean, minimalist {book['genre'].lower()} style",
                    f"Contemporary book cover featuring themes from '{book['title']}' - sleek typography and modern aesthetics",
                    f"Minimalist cover design for {book['genre']} book '{book['title']}' - professional and eye-catching",
                ],
                "vintage": [
                    f"Vintage-style book cover for '{book['title']}' - classic {book['genre'].lower()} design with retro typography",
                    f"Retro book cover design for '{book['title']}' by {book['author']} - nostalgic and timeless",
                    f"Classic vintage cover for {book['genre']} novel '{book['title']}' - elegant and traditional",
                ],
                "dramatic": [
                    f"Dramatic book cover for '{book['title']}' - bold {book['genre'].lower()} imagery with striking visuals",
                    f"High-impact cover design for '{book['title']}' by {book['author']} - compelling and intense",
                    f"Bold dramatic cover for {book['genre']} book '{book['title']}' - attention-grabbing design",
                ],
            }

            selected_prompts = prompts.get(style, prompts["modern"])

            # Create variants
            prompts_data = []
            for i, prompt in enumerate(selected_prompts):
                prompts_data.append(
                    {
                        "Prompt #": f"#{i+1}",
                        "Image Prompt": prompt,
                        "Focus": "Template",
                        "Style": style.title(),
                        "Length": len(prompt),
                    }
                )

            results_text = f"""
            ## 🎨 Image Prompts Generated (Fallback Mode)
            
            **Book:** {book['title']} by {book['author']}
            **Genre:** {book['genre']}
            **Style:** {style.title()}
            **Dimensions:** {dimensions}
            
            ### 📊 Summary
            - **Prompts Generated:** {len(prompts_data)}
            - **Method:** Template-based (fallback)
            """

            prompts_df = pd.DataFrame(prompts_data)
            return results_text, prompts_df

        except Exception as e:
            return f"Fallback image generation error: {str(e)}", pd.DataFrame()

    def generate_image_prompts_ui(self, book_selection, style, dimensions):
        """Generate image prompts for selected book"""
        try:
            book_id = self.get_book_id_from_selection(book_selection)

            # Try API first
            try:
                response = requests.post(
                    f"{self.api_base_url}/generate/image-prompts",
                    json={"book_id": book_id, "style": style},
                    timeout=10,
                )

                if response.status_code == 200:
                    data = response.json()
                    return self.format_image_prompts_results(data, dimensions)

            except requests.exceptions.RequestException:
                pass

            # Fallback generation
            return self.fallback_image_prompts_generation(book_id, style, dimensions)

        except Exception as e:
            return f"Error generating image prompts: {str(e)}", pd.DataFrame()

    def format_image_prompts_results(self, api_data, requested_dimensions):
        """Format image prompt results"""
        if "error" in api_data:
            return f"Error: {api_data['error']}", pd.DataFrame()

        results_text = f"""
        ## 🎨 Image Prompts Generated
        
        **Book ID:** {api_data.get('book_id', 'Unknown')}
        **Generated:** {api_data.get('generated_at', 'Unknown')}
        **Recommended Dimensions:** {api_data.get('recommended_dimensions', {}).get('cover', 'Unknown')}
        **Color Palette:** {', '.join(api_data.get('color_palette', []))}
        
        ### 🎯 Prompt Variants
        """

        # Create DataFrame for prompts
        prompts_data = []
        for i, prompt_data in enumerate(api_data.get("image_prompts", [])):
            prompts_data.append(
                {
                    "Prompt #": f"#{i+1}",
                    "Image Prompt": prompt_data.get("prompt", ""),
                    "Focus": prompt_data.get("focus", "").title(),
                    "Style": prompt_data.get("style", "").title(),
                    "Length": len(prompt_data.get("prompt", "")),
                }
            )

        prompts_df = pd.DataFrame(prompts_data)
        return results_text, prompts_df

    def get_inventory_insights(self, warehouse_filter):
        """Get inventory insights across warehouses"""
        try:
            if warehouse_filter == "All":
                warehouse_condition = ""
                params = []
            else:
                warehouse_condition = "WHERE i.warehouse_location = ?"
                params = [warehouse_filter]

            inventory_data = self.conn.execute(
                f"""
                SELECT 
                    i.warehouse_location,
                    b.genre,
                    COUNT(*) as unique_books,
                    SUM(i.stock_quantity) as total_stock,
                    AVG(i.stock_quantity) as avg_stock_per_book,
                    SUM(i.stock_quantity * b.price) as inventory_value
                FROM stg_inventory i
                JOIN dim_books b ON i.book_id = b.book_id
                {warehouse_condition}
                GROUP BY i.warehouse_location, b.genre
                ORDER BY inventory_value DESC
            """,
                params,
            ).df()

            if len(inventory_data) == 0:
                return "No inventory data found", go.Figure(), pd.DataFrame()

            # Create inventory visualization
            fig = px.treemap(
                inventory_data,
                path=["warehouse_location", "genre"],
                values="inventory_value",
                title="Inventory Value by Warehouse and Genre",
                height=500,
            )

            # Create summary
            total_value = inventory_data["inventory_value"].sum()
            total_books = inventory_data["unique_books"].sum()
            total_stock = inventory_data["total_stock"].sum()

            insights_text = f"""
            ## Inventory Insights
            
            ### Summary Statistics
            - **Total Inventory Value:** ${total_value:,.2f}
            - **Unique Books:** {total_books:,}
            - **Total Stock Units:** {total_stock:,}
            - **Warehouses:** {inventory_data['warehouse_location'].nunique()}
            - **Genres:** {inventory_data['genre'].nunique()}
            
            ### Top Performing Categories
            """

            top_categories = inventory_data.nlargest(5, "inventory_value")
            for _, row in top_categories.iterrows():
                insights_text += f"\n- **{row['warehouse_location']} - {row['genre']}:** ${row['inventory_value']:,.2f}"

            # Prepare display dataframe
            display_df = inventory_data.copy()
            display_df["inventory_value"] = display_df["inventory_value"].round(2)
            display_df["avg_stock_per_book"] = display_df["avg_stock_per_book"].round(1)
            display_df.columns = [
                "Warehouse",
                "Genre",
                "Unique Books",
                "Total Stock",
                "Avg Stock/Book",
                "Inventory Value ($)",
            ]

            return insights_text, fig, display_df

        except Exception as e:
            error_msg = f"Error getting inventory insights: {str(e)}"
            return error_msg, go.Figure(), pd.DataFrame()


def create_gradio_interface():
    """Create the main Gradio interface"""
    app = SomaGradioApp()

    # Get initial data for dropdowns
    book_list = app.get_book_list()
    warehouse_list = ["All"] + [
        "US_EAST",
        "US_WEST",
        "US_CENTRAL",
        "EU_LONDON",
        "EU_FRANKFURT",
        "ASIA_SINGAPORE",
    ]
    user_types = ["Any", "Individual", "Library", "School", "Bookstore"]

    with gr.Blocks(
        title="Soma - ML Analytics Platform",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .tab-nav {
            background: linear-gradient(90deg, #1e3a8a, #3b82f6);
        }
        """,
    ) as interface:

        gr.HTML(
            """
        <div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #1e3a8a, #3b82f6); color: white; border-radius: 10px; margin-bottom: 20px;">
            <h1>🚀 Soma - ML Analytics Platform</h1>
            <p>Advanced book demand forecasting, recommendations, and inventory optimization</p>
        </div>
        """
        )

        with gr.Tabs():
            # Tab 1: Demand Forecasting
            with gr.Tab("📈 Demand Forecasting"):
                gr.HTML("<h2>Predict future book demand using ML models</h2>")

                with gr.Row():
                    with gr.Column(scale=1):
                        book_dropdown_demand = gr.Dropdown(
                            choices=book_list,
                            label="Select Book",
                            value=book_list[0] if book_list else None,
                            interactive=True,
                        )
                        days_ahead = gr.Slider(
                            minimum=1,
                            maximum=30,
                            value=7,
                            step=1,
                            label="Forecast Days Ahead",
                        )

                        with gr.Accordion("Model Parameters", open=False):
                            confidence_level = gr.Slider(
                                minimum=0.8,
                                maximum=0.99,
                                value=0.95,
                                label="Confidence Level",
                            )
                            seasonality = gr.Checkbox(
                                label="Include Seasonality", value=True
                            )

                        predict_btn = gr.Button(
                            "🔮 Generate Forecast", variant="primary", size="lg"
                        )

                    with gr.Column(scale=2):
                        demand_results = gr.Markdown()
                        demand_plot = gr.Plot()
                        demand_table = gr.Dataframe(
                            headers=[
                                "Day",
                                "Predicted Demand",
                                "Lower Bound",
                                "Upper Bound",
                            ],
                            interactive=False,
                        )

                predict_btn.click(
                    fn=app.predict_demand,
                    inputs=[book_dropdown_demand, days_ahead, confidence_level],
                    outputs=[demand_results, demand_plot, demand_table],
                )

            # Tab 2: Book Recommendations
            with gr.Tab("🎯 Book Recommendations"):
                gr.HTML("<h2>Get personalized book recommendations</h2>")

                with gr.Row():
                    with gr.Column(scale=1):
                        book_dropdown_rec = gr.Dropdown(
                            choices=[""] + book_list,
                            label="Select Book (for similar recommendations)",
                            value="",
                            interactive=True,
                        )
                        user_type_dropdown = gr.Dropdown(
                            choices=user_types,
                            label="User Type",
                            value="Any",
                            interactive=True,
                        )
                        num_recommendations = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=10,
                            step=1,
                            label="Number of Recommendations",
                        )

                        recommend_btn = gr.Button(
                            "✨ Get Recommendations", variant="primary", size="lg"
                        )

                    with gr.Column(scale=2):
                        rec_results = gr.Markdown()
                        rec_table = gr.Dataframe(
                            headers=["Title", "Genre", "Confidence Score", "Reason"],
                            interactive=False,
                        )

                recommend_btn.click(
                    fn=app.get_recommendations,
                    inputs=[book_dropdown_rec, user_type_dropdown, num_recommendations],
                    outputs=[rec_results, rec_table],
                )

            # Tab 3: Book Analytics
            with gr.Tab("📊 Book Analytics"):
                gr.HTML("<h2>Comprehensive book performance analytics</h2>")

                with gr.Row():
                    with gr.Column(scale=1):
                        book_dropdown_analytics = gr.Dropdown(
                            choices=book_list,
                            label="Select Book for Analysis",
                            value=book_list[0] if book_list else None,
                            interactive=True,
                        )

                        analytics_btn = gr.Button(
                            "📊 Generate Analytics", variant="primary", size="lg"
                        )

                    with gr.Column(scale=2):
                        analytics_results = gr.Markdown()
                        analytics_plot = gr.Plot()
                        analytics_table = gr.Dataframe(
                            headers=[
                                "Month",
                                "Units Sold",
                                "Revenue",
                                "Transactions",
                                "Avg Price",
                            ],
                            interactive=False,
                        )

                analytics_btn.click(
                    fn=app.get_book_analytics,
                    inputs=[book_dropdown_analytics],
                    outputs=[analytics_results, analytics_plot, analytics_table],
                )

            # Tab 4: Inventory Insights
            with gr.Tab("📦 Inventory Insights"):
                gr.HTML("<h2>Warehouse inventory optimization insights</h2>")

                with gr.Row():
                    with gr.Column(scale=1):
                        warehouse_dropdown = gr.Dropdown(
                            choices=warehouse_list,
                            label="Select Warehouse",
                            value="All",
                            interactive=True,
                        )

                        inventory_btn = gr.Button(
                            "📦 Analyze Inventory", variant="primary", size="lg"
                        )

                    with gr.Column(scale=2):
                        inventory_results = gr.Markdown()
                        inventory_plot = gr.Plot()
                        inventory_table = gr.Dataframe(
                            headers=[
                                "Warehouse",
                                "Genre",
                                "Books",
                                "Stock",
                                "Avg Stock",
                                "Value",
                            ],
                            interactive=False,
                        )

                inventory_btn.click(
                    fn=app.get_inventory_insights,
                    inputs=[warehouse_dropdown],
                    outputs=[inventory_results, inventory_plot, inventory_table],
                )

            # Tab 5: Ad Copy Generation
            with gr.Tab("🎯 Ad Copy Generation"):
                gr.HTML("<h2>Generate compelling ad copy using RAG-powered AI</h2>")

                with gr.Row():
                    with gr.Column(scale=1):
                        book_dropdown_ads = gr.Dropdown(
                            choices=book_list,
                            label="Select Book for Ad Copy",
                            value=book_list[0] if book_list else None,
                            interactive=True,
                        )
                        ad_type_dropdown = gr.Dropdown(
                            choices=["social_media", "email", "display", "print"],
                            label="Ad Type",
                            value="social_media",
                            interactive=True,
                        )
                        target_audience_dropdown = gr.Dropdown(
                            choices=[
                                "Auto-detect",
                                "Young Adult",
                                "Professional",
                                "Academic",
                                "General",
                                "Children",
                            ],
                            label="Target Audience",
                            value="Auto-detect",
                            interactive=True,
                        )
                        style_preference = gr.Dropdown(
                            choices=[
                                "Creative",
                                "Professional",
                                "Casual",
                                "Urgent",
                                "Elegant",
                            ],
                            label="Style Preference",
                            value="Creative",
                            interactive=True,
                        )

                        generate_ads_btn = gr.Button(
                            "🚀 Generate Ad Copy", variant="primary", size="lg"
                        )

                    with gr.Column(scale=2):
                        ad_copy_results = gr.Markdown()
                        ad_copy_table = gr.Dataframe(
                            headers=[
                                "Variant",
                                "Ad Copy",
                                "Type",
                                "Confidence",
                                "Length",
                                "Call to Action",
                            ],
                            interactive=False,
                            wrap=True,
                        )

                generate_ads_btn.click(
                    fn=app.generate_ad_copy_ui,
                    inputs=[
                        book_dropdown_ads,
                        ad_type_dropdown,
                        target_audience_dropdown,
                        style_preference,
                    ],
                    outputs=[ad_copy_results, ad_copy_table],
                )

            # Tab 6: Image Prompt Generation
            with gr.Tab("🎨 Image Prompts"):
                gr.HTML(
                    "<h2>Generate AI image prompts for book covers and marketing</h2>"
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        book_dropdown_images = gr.Dropdown(
                            choices=book_list,
                            label="Select Book for Image Prompts",
                            value=book_list[0] if book_list else None,
                            interactive=True,
                        )
                        image_style = gr.Dropdown(
                            choices=[
                                "modern",
                                "vintage",
                                "minimalist",
                                "dramatic",
                                "artistic",
                                "commercial",
                            ],
                            label="Image Style",
                            value="modern",
                            interactive=True,
                        )
                        image_dimensions = gr.Dropdown(
                            choices=[
                                "1600x2400 (Book Cover)",
                                "1080x1080 (Social Square)",
                                "1200x628 (Social Banner)",
                                "Custom",
                            ],
                            label="Target Dimensions",
                            value="1600x2400 (Book Cover)",
                            interactive=True,
                        )

                        generate_images_btn = gr.Button(
                            "🎨 Generate Image Prompts", variant="primary", size="lg"
                        )

                    with gr.Column(scale=2):
                        image_prompts_results = gr.Markdown()
                        image_prompts_table = gr.Dataframe(
                            headers=[
                                "Prompt #",
                                "Image Prompt",
                                "Focus",
                                "Style",
                                "Length",
                            ],
                            interactive=False,
                            wrap=True,
                        )

                generate_images_btn.click(
                    fn=app.generate_image_prompts_ui,
                    inputs=[book_dropdown_images, image_style, image_dimensions],
                    outputs=[image_prompts_results, image_prompts_table],
                )

            # Tab 7: System Status
            with gr.Tab("⚙️ System Status"):
                gr.HTML("<h2>System health and model status</h2>")

                def get_system_status():
                    """Get comprehensive system status"""
                    status_text = "## System Status\n\n"

                    # Database status
                    if app.conn:
                        try:
                            book_count = app.conn.execute(
                                "SELECT COUNT(*) FROM dim_books"
                            ).fetchone()[0]
                            sales_count = app.conn.execute(
                                "SELECT COUNT(*) FROM fact_sales"
                            ).fetchone()[0]
                            status_text += f"✅ **Database:** Connected\n"
                            status_text += f"📚 **Books:** {book_count:,}\n"
                            status_text += f"💰 **Sales Records:** {sales_count:,}\n\n"
                        except Exception as e:
                            status_text += f"❌ **Database Error:** {e}\n\n"
                    else:
                        status_text += "❌ **Database:** Not connected\n\n"

                    # API status
                    try:
                        response = requests.get(f"{app.api_base_url}/health", timeout=5)
                        if response.status_code == 200:
                            health_data = response.json()
                            status_text += "✅ **ML API:** Connected\n"
                            status_text += f"🤖 **Models Loaded:** {', '.join(health_data.get('models_loaded', []))}\n"
                        else:
                            status_text += (
                                f"⚠️ **ML API:** HTTP {response.status_code}\n"
                            )
                    except:
                        status_text += (
                            "❌ **ML API:** Not available (using fallback mode)\n"
                        )

                    status_text += f"\n🕐 **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

                    return status_text

                status_btn = gr.Button("🔄 Refresh Status", variant="secondary")
                status_output = gr.Markdown()

                status_btn.click(fn=get_system_status, outputs=[status_output])

                # Load initial status
                interface.load(fn=get_system_status, outputs=[status_output])

        gr.HTML(
            """
        <div style="text-align: center; padding: 10px; margin-top: 20px; color: #666;">
            <p>Soma ML Platform v1.0 | Built with ❤️ using Gradio</p>
        </div>
        """
        )

    return interface


if __name__ == "__main__":
    print("🚀 Starting Soma ML Platform...")
    print("📊 Initializing Gradio interface...")

    # Create and launch interface
    interface = create_gradio_interface()

    print("✅ Interface ready!")
    print("🌐 Launching web application...")

    # Launch with appropriate settings
    interface.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,  # Standard Gradio port
        share=False,  # Set to True for public sharing
        debug=True,  # Enable debug mode
        show_error=True,  # Show detailed error messages
        quiet=False,  # Show startup logs
    )
