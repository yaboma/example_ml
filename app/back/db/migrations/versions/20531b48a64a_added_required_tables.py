"""Added required tables

Revision ID: 20531b48a64a
Revises:
Create Date: 2022-05-18 19:14:35.802270

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20531b48a64a"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table(
        "mas_predicted",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("nrot", sa.String(length=9), nullable=False),
        sa.Column("dt", sa.DateTime(), nullable=True),
        sa.Column("num_experiments", sa.Integer(), nullable=True),
        sa.Column("mas_val_up", sa.Integer(), nullable=True),
        sa.Column("mas_ang_up", sa.Integer(), nullable=True),
        sa.Column("mas_val_down", sa.Integer(), nullable=True),
        sa.Column("mas_ang_down", sa.Integer(), nullable=True),
        sa.Column("mas_val_middle", sa.Integer(), nullable=True),
        sa.Column("mas_ang_middle", sa.Integer(), nullable=True),
        sa.Column("mas_val_middle_up", sa.Integer(), nullable=True),
        sa.Column("mas_ang_middle_up", sa.Integer(), nullable=True),
        sa.Column("mas_val_middle_down", sa.Integer(), nullable=True),
        sa.Column("mas_ang_middle_down", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_mas_predicted_dt"), "mas_predicted", ["dt"], unique=False
    )
    op.create_index(
        op.f("ix_mas_predicted_nrot"), "mas_predicted", ["nrot"], unique=False
    )
    op.create_index(
        op.f("ix_mas_predicted_num_experiments"),
        "mas_predicted",
        ["num_experiments"],
        unique=False,
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(
        op.f("ix_mas_predicted_num_experiments"), table_name="mas_predicted"
    )
    op.drop_index(op.f("ix_mas_predicted_nrot"), table_name="mas_predicted")
    op.drop_index(op.f("ix_mas_predicted_dt"), table_name="mas_predicted")
    op.drop_table("mas_predicted")
    # ### end Alembic commands ###