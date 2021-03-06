"""create prediction tables

Revision ID: 1000000
Revises: 
Create Date: 2020-05-31 14:54:07.857500+00:00

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "1000000"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table(
        "cvd_model_predictions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column(
            "datetime_captured",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column("model_version", sa.String(length=36), nullable=False),
        sa.Column("inputs", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("outputs", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_cvd_model_predictions_datetime_captured"),
        "cvd_model_predictions",
        ["datetime_captured"],
        unique=False,
    )
    
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(
        op.f("ix_cvd_model_predictions_datetime_captured"),
        table_name="cvd_model_predictions",
    )
    op.drop_table("cvd_model_predictions")
    # ### end Alembic commands ###
