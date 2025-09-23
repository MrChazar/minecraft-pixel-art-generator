/*
  Warnings:

  - You are about to drop the column `user_id` on the `Image` table. All the data in the column will be lost.

*/
-- DropForeignKey
ALTER TABLE "public"."Image" DROP CONSTRAINT "Image_user_id_fkey";

-- AlterTable
ALTER TABLE "public"."Image" DROP COLUMN "user_id";
